"""For mapping food labels with their calorie and macro data with USDA FoodData Central API"""
import logging

import requests
import json
import csv
import os
import asyncio
import aiohttp
from redis import Redis, ConnectionError as RedisConnectionError
from typing import Optional, Dict



class NutritionMapper:
    def __init__(self, api_key:str,redis_url: str = "redis://localhost:6379",
                 cache_file:str="nutrition_cache.csv"):
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

    def _verify_redis_connection(self):
        try:
            self.redis.ping()
        except RedisConnectionError:
            logging.warning("Warning: Redis connection failed. Using in-memory cache")
            self.redis=None
        except Exception as e:
            logging.error(f"Unexpected Redis error: {e}")
            self.redis=None
    async def close(self):
        await self.session.close()
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    async def get_nutrition_data(self, query: str, max_results: int = 1) -> Optional[Dict]:
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
                    nutrition = self._parse_nutrition_data(data["foods"][0])
                    self._cache_data(cache_key, nutrition)
                    return nutrition
              return None

    def _validate_nutrition_values(self, nutrition: Dict) -> Dict:
        """Validate and clean nutrition values"""
        cleaned = {}
        for key, value in nutrition.items():
            try:
                cleaned[key] = float(value)
                if cleaned[key] < 0:
                    cleaned[key] = 0
                    logging.warning(f"Negative value found for {key}, set to 0")
            except (ValueError, TypeError):
                logging.warning(f"Invalid value for {key}: {value}")
                cleaned[key] = 0
        return cleaned

    async def _api_call_with_retry(self, url: str, params: Dict, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:  # Rate limit
                        await asyncio.sleep(2 ** attempt)
                        continue
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        return None

    def _parse_nutrition_data(self, food_item: Dict) -> Dict:
        """Parse nutrition data from USDA API response"""
        nutrition = {
            "calories": 0,
            "protein": 0,
            "fat": 0,
            "carbohydrates": 0
        }

        for nutrient in food_item.get("foodNutrients", []):
            name = nutrient.get("nutrientName", "").lower()
            value = nutrient.get("value", 0)

            if "calorie" in name:
                nutrition["calories"] = value
            elif "protein" in name:
                nutrition["protein"] = value
            elif "fat" in name:
                nutrition["fat"] = value
            elif "carbohydrate" in name:
                nutrition["carbohydrates"] = value

        return nutrition

    def _cache_data(self, key: str, nutrition: Dict):
        """Cache data in both Redis and local cache"""
        self.cache[key.split(":")[1]] = nutrition

        if self.redis:
            try:
                # Store with 24-hour TTL
                self.redis.hset(key, mapping=nutrition)
                self.redis.expire(key, 86400)
            except RedisConnectionError:
                pass

    async def map_food_label_to_nutrition(self, food_label: str) -> Dict:
        """Async version with caching and fallback"""
        try:
            nutrition = await self.get_nutrition_data(food_label)

            if not nutrition:
                return self.get_default_nutrition()

            density = self.density_db.get(food_label.lower(), 0.05)
            nutrition["calories_per_ml"] = (nutrition["calories"] / 100) * density

            return nutrition
        except Exception as e:
            logging.error(f"Nutrition mapping failed for {food_label}: {e}")
            return self.get_default_nutrition()

    def get_density(self, food_name:str)->float:
        return self.density_db.get(food_name.lower(), 0.05)

    def _load_cache(self):
        """Load cache nutrition data from a CSV file, if available"""
        cache = {}
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    label = row["label"]
                    cache[label.lower()] = {"calories":float(row["calories"]),
                                            "protein":float(row["protein"]),
                                            "fat":float(row["fat"]),
                                            "carbohydrates":float(row["carbohydrates"])}
        return cache

    def get_default_nutrition(self):
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
    nutrition = mapper.map_food_label_to_nutrition(food_label)
    if nutrition:
        print(f"Nutrition data for '{food_label}':")
        print(f"Calories: {nutrition['calories']} kcal")
        print(f"Protein: {nutrition['protein']} g")
        print(f"Fat: {nutrition['fat']} g")
        print(f"Carbohydrates: {nutrition['carbohydrates']} g")
    else:
        print(f"No nutrition data found for '{food_label}'.")