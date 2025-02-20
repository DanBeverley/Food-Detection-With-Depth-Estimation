import logging
import csv
import os
import asyncio
import socket

import aiohttp
from redis import Redis, ConnectionError as RedisConnectionError
from typing import Optional, Dict, Any, cast


class NutritionMapper:
    def __init__(self, api_key:str,redis_url: str = "redis://localhost:6379",
                 cache_file:str="nutrition_cache.csv") -> None:
        """
        Initializes the NutritionWrapper
        Args:
            api_key(str): USDA FoodData Central API key
            cache_file(str): Optional CSV file path to cache results
        """
        self.api_key = api_key
        self.cache_file = cache_file
        self.base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self._verify_redis_connection()
        self.cache = self._load_cache()
        self.density_db = {
            "rice": 0.05, "white rice": 0.05, "brown rice": 0.05, "bread": 0.03,
            "whole wheat bread": 0.03, "meat": 0.08, "chicken": 0.08, "beef": 0.09,
            "pork": 0.08, "fish": 0.08, "pasta": 0.06, "potato": 0.07, "vegetables": 0.04,
            "broccoli": 0.04, "carrot": 0.04, "salad": 0.02, "fruit": 0.04, "apple": 0.04,
            "banana": 0.04, "orange": 0.04, "cereal": 0.03, "oatmeal": 0.05, "soup": 0.09,
            "stew": 0.09, "cheese": 0.1, "egg": 0.09, "burger": 0.12, "pizza": 0.1, "taco": 0.07,
            "pancakes": 0.04, "waffles": 0.04, "cookies": 0.03, "brownie": 0.03, "cake": 0.03,
            "pie": 0.03, "yogurt": 0.05, "ice cream": 0.06, "almonds": 0.07, "peanuts": 0.07
        }
        self.session = aiohttp.ClientSession()

    def _verify_redis_connection(self) -> None:
        try:
            self.redis.ping()
        except RedisConnectionError:
            logging.warning("Warning: Redis connection failed. Using in-memory cache")
            self.redis=None
        except Exception as e:
            logging.error(f"Unexpected Redis error: {e}")
            self.redis=None
    async def close(self) -> None:
        await self.session.close()
    async def __aenter__(self) -> "NutritionMapper":
        return self
    async def __aexit__(self, exc_type: Any, exc_val:Any, exc_tb:Any) -> None:
        await self.close()
    async def get_nutrition_data(self, query: str, max_results: int = 1) -> Optional[Dict[str, float]]:
        """Async version of the USDA API call with Redis caching"""
        cache_key = f"nutrition:{query.lower()}"

        # Try Redis cache first
        if self.redis:
            try:
                cached = self.redis.hgetall(cache_key)
                if cached:
                    return {k: float(v) for k, v in cached.items()}
            except RedisConnectionError:
                pass

        # Try local cache
        if query.lower() in self.cache:
            return self.cache[query.lower()]

        # API call with retry logic
        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": max_results
        }
        async with self.session.get(self.base_url, params=params) as response:
              if response.status == 200:
                data = await response.json()
                if "foods" in data and len(data["foods"])>0:
                    nutritions = self._parse_nutrition_data(data["foods"][0])
                    self._cache_data(cache_key, nutritions)
                    return nutritions
              return None

    @staticmethod
    def _validate_nutrition_values(nutritions: Dict) -> Dict:
        """Validate and clean nutrition values"""
        cleaned = {}
        for key, value in nutritions.items():
            try:
                cleaned[key] = float(value)
                if cleaned[key] < 0:
                    cleaned[key] = 0
                    logging.warning(f"Negative value found for {key}, set to 0")
            except (ValueError, TypeError):
                logging.warning(f"Invalid value for {key}: {value}")
                cleaned[key] = 0
        return cleaned

    async def _api_call_with_retry(self, url: str, params: Dict[str, Any],
                                   max_retries: int = 3) -> Optional[Dict[str, Any]]:
        for attempt in range(max_retries):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        await asyncio.sleep(2 ** attempt)
                        continue
            except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError, socket.gaierror, asyncio.CancelledError) as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
                logging.warning(f"Request failed: {e}. Retrying (attempt {attempt+1}/{max_retries})...")
            except Exception as e:
                logging.warning(f"Unexpected error occured: {e}")
                raise
        return None

    @staticmethod
    def _parse_nutrition_data(food_item: Dict[str, Any]) -> Dict[str, float]:
        """Parse nutrition data from USDA API response"""
        nutritions = {
            "calories": 0,
            "protein": 0,
            "fat": 0,
            "carbohydrates": 0
        }

        for nutrient in food_item.get("foodNutrients", []):
            name = nutrient.get("nutrientName", "").lower()
            value = nutrient.get("value", 0)

            if "calorie" in name:
                nutritions["calories"] = value
            elif "protein" in name:
                nutritions["protein"] = value
            elif "fat" in name:
                nutritions["fat"] = value
            elif "carbohydrate" in name:
                nutritions["carbohydrates"] = value

        return nutritions

    def _cache_data(self, key: str, nutritions: Dict[str, float]) -> None:
        """Cache data in both Redis and local cache"""
        self.cache[key.split(":")[1]] = nutritions

        if self.redis is not None:
            try:
                # Store with 24-hour TTL
                self.redis.hset(key, mapping=nutritions)
                self.redis.expire(key, 86400)
            except RedisConnectionError:
                pass

    async def map_food_label_to_nutrition(self, food_labels: str) -> Dict[str, float]:
        """Async version with caching and fallback"""
        session = getattr(self, "session", aiohttp.ClientSession())
        try:
            params = {"api_key": self.api_key, "query": food_labels}
            async with session.get(self.base_url, params=params) as response:
                  if response.status == 200:
                      data = await response.json()
                      return data
        finally:
            if not hasattr(self, "session"):
                await session.close()
        attempts = 3
        for attempt in range(attempts):
            try:
                nutritions = await self.get_nutrition_data(food_labels)
                if not nutritions:
                    logging.warning(f"No nutrition data found for {food_labels}, using default nutrition")
                    return self.get_default_nutrition()
                density = self.density_db.get(food_labels.lower(), 0.05)
                nutritions["calories_per_ml"] = (nutritions["calories"] / 100) * density
                return nutritions
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == attempts - 1:
                    logging.error(f"Failed after {attempts} attempts: {e}, Switching to use default nutrition")
                    return self.get_default_nutrition()
                await asyncio.sleep(2 ** attempt)

    def get_density(self, food_name:str)->float:
        return self.density_db.get(food_name.lower(), 0.05)

    def _load_cache(self) -> Dict[str, Dict[str, float]]:
        """Load cache nutrition data from a CSV file, if available"""
        cache = {}
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    row_dict = cast(Dict[str, str], row)
                    label = row_dict["label"]
                    cache[label.lower()] = {"calories":float(row_dict["calories"]),
                                            "protein":float(row_dict["protein"]),
                                            "fat":float(row_dict["fat"]),
                                            "carbohydrates":float(row_dict["carbohydrates"])}
        return cache

    @staticmethod
    def get_default_nutrition() -> Dict[str, float]:
        """
        Returns default nutrional data as a dictionary
        Used as a fallback when no nutrion data is available
        """
        return {"calories":0,
                "protein":0,
                "fat":0,
                "carbohydrates":0,
                "calories_per_ml":.5}

if __name__ == "__main__":
    API_KEY = "API_KEY"
    mapper = NutritionMapper(api_key=API_KEY)

    # Example: Map the food label 'rice' to its nutrition data.
    food_label = "rice"
    nutrition = asyncio.run(mapper.map_food_label_to_nutrition(food_label))
    if nutrition:
        print(f"Nutrition data for '{food_label}':")
        print(f"Calories: {nutrition['calories']} kcal")
        print(f"Protein: {nutrition['protein']} g")
        print(f"Fat: {nutrition['fat']} g")
        print(f"Carbohydrates: {nutrition['carbohydrates']} g")
    else:
        print(f"No nutrition data found for '{food_label}'.")