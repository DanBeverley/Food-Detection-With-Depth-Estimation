import json
import logging
import csv
import os
import asyncio
import aiohttp
from redis import Redis, ConnectionError as RedisConnectionError
from typing import Dict, Any, cast

from tenacity import retry
from tenacity import wait_exponential, stop_after_attempt, retry_if_exception_type


class TooManyRequestsError(Exception):
    pass

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
        self._request_semaphore = asyncio.Semaphore(5)
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
        if self.redis is None:
            logging.warning("Redis connection is not initialized. Using in memory-cache")
            return
        @retry(wait = wait_exponential(multiplier=0.1, min=0.1, max=1),
               stop = stop_after_attempt(3), retry = retry_if_exception_type(RedisConnectionError),
               reraise = True)
        def ping_redis():
            self.redis.ping()
        try:
            ping_redis()
        except RedisConnectionError:
            logging.warning("⚠️ Redis connection failed after retries. Using in-memory cache.")
            self.redis = None
        except Exception as e:
            logging.error(f"Unexpected Redis error: {e}")
            self.redis = None

    async def close(self) -> None:
        await self.session.close()
    async def __aenter__(self) -> "NutritionMapper":
        return self
    async def __aexit__(self, exc_type: Any, exc_val:Any, exc_tb:Any) -> None:
        await self.close()
    async def get_nutrition_data(self, queries: str, max_concurrent: int = 5) -> Dict[str, Dict[str, float]]:
        """Async version of the USDA API call with Redis caching"""
        async with self._request_semaphore:
            results = {}
            missing_queries = []

            # Deduplicate queries
            unique_queries = set(queries)
            # Check cache
            for query in unique_queries:
                cache_key = f"Nutrition: {query.lower()}"
                if query.lower() in self.cache:
                    results[query] = self.cache[query.lower()]
                    continue
                # Redis after local cache
                if self.redis:
                    try:
                        cached = self.redis.get(cache_key)
                        if cached:
                            results[query] = json.loads(cached)
                            continue
                    except RedisConnectionError:
                        logging.warning("Redis connection failed")
                missing_queries.append(query)
            # Fetch missing data with concurrency limit
            semaphore = asyncio.Semaphore(max_concurrent)
            async def fetch_with_limit(query):
                async with semaphore:
                    return await self.get_nutrition_data(query)
            if missing_queries:
                tasks = [fetch_with_limit(query) for query in missing_queries]
                fetched_results = await asyncio.gather(*tasks, return_exceptions=True)
                for query, result in zip(missing_queries, fetched_results):
                    if isinstance(result, Exception):
                        logging.error(f"Error fetching nutrition for {query}:{result}")
                        results[query] = self.get_nutrition_data()
                    else:
                        results[query] = result
            return results

    @staticmethod
    def _parse_nutrition_data(food_item: Dict[str, Any]) -> Dict[str, float]:
        """Parse nutrition data from USDA API response"""
        NUTRITION_IDS = {"calories":1008,
                         "protein":1003,
                         "fat":1004,
                         "carbohydrates":1005}
        nutritions = {
            "calories": 0,
            "protein": 0,
            "fat": 0,
            "carbohydrates": 0
        }
        for nutrient in food_item.get("foodNutrients", []):
            nutrient_id = nutrient.get("nutrientId")
            amount = nutrient.get("value", 0)
            if nutrient_id == NUTRITION_IDS["calories"]:
                nutritions["calories"] = float(amount)
            elif nutrient_id == NUTRITION_IDS["protein"]:
                nutritions["protein"] = float(amount)
            elif nutrient_id == NUTRITION_IDS["fat"]:
                nutritions["fat"] = float(amount)
            elif nutrient_id == NUTRITION_IDS["carbohydrates"]:
                nutritions["carbohydrates"] = float(amount)
        if sum(nutritions.values())==0:
            logging.warning(f"No nutrition data found for food item: {food_item.get('description')}")
            return NutritionMapper.get_default_nutrition()
        return nutritions

    def _cache_data(self, key: str, nutritions: Dict[str, float]) -> None:
        """Cache data in both Redis and local cache"""
        self.cache[key.lower()] = nutritions
        if self.redis is not None:
            try:
                # Store with 24-hour TTL
                cache_key = f"nutrition:{key.lower()}"
                self.redis.set(cache_key, json.dumps(nutritions))
                self.redis.expire(cache_key, 86400)
            except RedisConnectionError:
                pass

    async def map_food_label_to_nutrition(self, food_labels: str) -> Dict[str, float]:
        """Async version with caching and fallback"""
        nutritions = await self.get_nutrition_data(food_labels)
        if not nutritions:
          logging.warning(f"No nutrition data found for {food_labels}, using default nutrition")
          return self.get_default_nutrition()
        density = self.density_db.get(food_labels.lower(), 0.05)
        if not isinstance(density, (int, float)):
            density = 0.05 # Fallback
        try:
            calories_per_ml = (nutritions["calories"] / 100.0) * density
        except KeyError:
            logging.error(f"Missing 'calories' key in nutritionm for {food_labels}")
            calories_per_ml = 0.0
        nutritions["calories_per_ml"] = float(calories_per_ml)
        return nutritions

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
