import json
import logging
import csv
import os
import asyncio
import aiohttp
from redis import Redis, ConnectionError as RedisConnectionError
from typing import Dict, Any, cast, Optional, List

from tenacity import retry
from tenacity import wait_exponential, stop_after_attempt, retry_if_exception_type


class TooManyRequestsError(Exception):
    pass

class NutritionMapper:
    def __init__(self, api_key:str,redis_url: str = "redis://localhost:6379",
                 cache_file:str="nutrition_cache.csv") -> None:
        # TODO: adjust both api_key and cache_file before cloud GPU usage . Also set up REDIS
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

    async def close(self) -> None:
        await self.session.close()

    async def __aenter__(self) -> "NutritionMapper":
        return self

    async def __aexit__(self, exc_type: Any, exc_val:Any, exc_tb:Any) -> None:
        await self.close()

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    async def _fetch_from_api(self, query:str)->Dict[str, float]:
        """Fetch nutrition data from USDA API for a single query"""
        params = {"api_key": self.api_key, "query":query, "pageSize":1}
        try:
            async with self.session.get(self.base_url, params = params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "foods" in data and data["foods"]:
                        return self._parse_nutrition_data(data["foods"][0])
                    logging.warning(f"No nutrition data found for {query}")
                    return self.get_default_nutrition()
                elif response.status == 429:
                    retry_after = response.headers.get("Retry-After", 1)
                    logging.warning(f"HTTP 429: Too Many Requests. Retry - After: {retry_after} sec")
                    raise TooManyRequestsError("Too many requests")
                else:
                    response.raise_for_status()
        except (aiohttp.ClientError, TooManyRequestsError) as e:
            logging.error(f"API request failed for {query}: {e}")
            return self.get_default_nutrition()
        return self.get_default_nutrition()

    async def _get_cached_data(self, query:str) -> Optional[Dict[str, float]]:
        """Check caches for nutrition data, return None if not found"""
        cache_key = f"nutrition:{query.lower()}"
        # Local cache
        if query.lower() in self.cache:
            return self.cache[query.lower()]
        # Redis cache
        if self.redis:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except RedisConnectionError:
                logging.warning("Redis connection failed")
        return None

    async def get_nutrition_data(self, query: str) -> Dict[str, float]:
        """Fetch nutrition data for a single query with caching"""
        cached = await self._get_cached_data(query)
        if cached is not None:
            return cached
        nutrition = await self._fetch_from_api(query)
        self._cache_data(query, nutrition)
        return nutrition

    async def batch_get_nutrition_data(self, queries: List[str], max_concurrent:int = 5) -> Dict[str, Dict[str, float]]:
        """Fetch nutrition data for multiple queries in parallel with caching"""
        results = {}
        missing_queries = []
        # Deduplicates and check caches
        for query in set(queries):
            cached = await self._get_cached_data(query)
            if cached is not None:
                results[query] = cached
            else:
                missing_queries.append(query)
        # Fetch missing queries in parallel
        if missing_queries:
            sem = asyncio.Semaphore(max_concurrent)

            async def fetch_with_limit(query: str) -> tuple[str, Dict[str, float]]:
                async with sem:
                    nutrition = await self.get_nutrition_data(query)
                    return query, nutrition
            tasks = [fetch_with_limit(query) for query in missing_queries]
            fetched = await asyncio.gather(*tasks, return_exceptions = True)
            for query, result in fetched:
                if isinstance(result, Exception):
                    logging.error(f"Failed to fetch nutrition data for {query}: {result}")
                    results[query] = self.get_default_nutrition()
                else:
                    results[query] = result
            results.update(dict(fetched))
        return results

    def get_cached_nutrition(self, query: str) -> Optional[Dict[str, float]]:
        """Get cached nutrition data synchronously"""
        if query.lower() in self.cache:
          return self.cache.get(query.lower())
        if self.redis:
            try:
                cache_key = f"nutrition:{query.lower()}"
                cached = self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logging.error(f"Redis error: {e}")
        logging.debug(f"No cached nutrition data for {query}, using default")
        return self.get_default_nutrition()

    @staticmethod
    def _parse_nutrition_data(food_item: Dict[str, Any]) -> Dict[str, float]:
        """Parse nutrition data from USDA API response"""
        NUTRITION_IDS = {"calories":1008, "protein":1003, "fat":1004, "carbohydrates":1005}
        nutritions = {"calories": 0, "protein": 0, "fat": 0, "carbohydrates": 0}
        nutrients_found = False
        for nutrient in food_item.get("foodNutrients", []):
            nutrient_id = nutrient.get("nutrientId")
            amount = float(nutrient.get("value", 0))
            if nutrient_id in NUTRITION_IDS.values():
                key = next(k for k, v in NUTRITION_IDS.items() if v == nutrient_id)
                nutritions[key] = amount
                nutrients_found = True
        if not nutrients_found:
            logging.warning(f"No nutrition data found for food item: {food_item.get('description')}")
            return NutritionMapper.get_default_nutrition()
        return nutritions

    def _cache_data(self, key: str, nutritions: Dict[str, float]) -> None:
        """Cache data in both Redis and local cache"""
        self.cache[key.lower()] = nutritions
        if self.redis:
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
        required_keys = ["calories", "protein", "fat", "carbohydrates"]
        if not all(k in nutritions for k in required_keys):
          logging.warning(f"No nutrition data found for {food_labels}, using default nutrition")
          return self.get_default_nutrition()
        density = self.density_db.get(food_labels.lower(), 0.05)
        if not isinstance(density, (int, float)):
            density = 0.05 # Fallback
        nutritions["calories_per_ml"] = (nutritions.get("calories", 0.0)/100.0) * density
        return nutritions

    def get_density(self, food_name:str)->float:
        return self.density_db.get(food_name.lower(), 0.05)

    def _load_cache(self) -> Dict[str, Dict[str, float]]:
        """Load cache nutrition data from a CSV file, if available"""
        cache = {}
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                required_keys = ["label", "calories", "protein", "fat", "carbohydrates"]
                if not required_keys.issubset(reader.fieldnames):
                    logging.error(f"CSV missing required columns: {required_keys - set(reader.fieldnames)}")
                    return cache
                for row in reader:
                    try:
                        label = row["label"].lower()
                        cache[label] = {k:float(row[k]) for k in required_keys[1:]}
                    except (KeyError, ValueError) as e:
                        logging.warning(f"Invalid cache entry: {e}")
        return cache

    @staticmethod
    def get_default_nutrition() -> Dict[str, float]:
        """
        Returns default nutrional data as a dictionary
        Used as a fallback when no nutrion data is available
        """
        return {"calories":0, "protein":0, "fat":0, "carbohydrates":0, "calories_per_ml":.5}
