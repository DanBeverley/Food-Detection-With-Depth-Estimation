"""For mapping food labels with their calorie and macro data with USDA FoodData Central API"""
import requests
import json
import csv
import os

class NutritionMapper:
    def __init__(self, api_key:str, cache_file:str="nutrition_cache.csv"):
        """
        Initializes the NutritionWrapper
        Args:
            api_key(str): USDA FoodData Central API key
            cache_file(str): Optional CSV file path to cache results
        """
        self.api_key = api_key
        self.cache_file = cache_file
        self.base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        self.cache = self._load_cache()
        self.density_db = {
            "rice":0.05,
            "bread":0.03,
            "meat":0.08 #TODO: Add more food type
        }
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

    def _save_cache_entry(self, label:str, nutrition_data:dict):
        """Append new cache entry to the CSV file"""
        file_exists = os.path.exists(self.cache_file)
        with open(self.cache_file, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["label", "calories", "protein", "fat", "carbohydrates"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "label":label,
                "calories":nutrition_data.get("calories", 0),
                "protein":nutrition_data.get("protein", 0),
                "fat":nutrition_data.get("fat", 0),
                "carbohydrates":nutrition_data.get("carbohydrates",0)})
    def get_nutrition_data(self, query:str, max_results:int=1):
        """
        Queries the USDA FoodData Central API for nutrition data.

        Args:
            query (str): Food item to search for.
            max_results (int): Maximum number of search results to return.

        Returns:
            dict or None: The first food item result from the API or None if not found.
        """
        params = {"api_key":self.api_key,
                  "query":query,
                  "pageSize":max_results}
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "foods" in data and len(data["foods"])>0:
                return data["foods"][0]
            else:
                return None
        else:
            print(f"Error fetching nutrition data: {response.status_code}")
            return None
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
    def map_food_label_to_nutrition(self, food_label:str):
        """
        Maps a food label to its nutrition data (calories, protein, fat, carbohydrates).

        Args:
            food_label (str): The food label (e.g., 'rice', 'chicken breast').

        Returns:
            dict or None: Dictionary with nutritional info or None if not found.
        """
        key = food_label.lower()
        if key in self.cache:
            return self.cache[key]
        nutrition_data = self.get_nutrition_data(food_label)
        if nutrition_data:
           # Extract nutrition values from the API response.
           # USDA API responses include a list of foodNutrients.
           calories = None
           protein = None
           fat = None
           carbohydrates = None
           for nutrient in nutrition_data.get("foodNutrients", []):
               nutrient_name = nutrient.get("nutrientName", "").lower()
               value = nutrient.get("value")
               if "energy" in nutrient_name or "calorie" in nutrient_name:
                   calories = value
               elif "protein" in nutrient_name:
                   protein = value
               elif "total lipid" in nutrient_name or "fat" in nutrient_name:
                   fat = value
               elif "carbohydrate" in nutrient_name:
                   carbohydrates = value
           result = {"calories":calories if calories is not None else 0,
                     "protein":protein if protein is not None else 0,
                     "fat":fat if fat is not None else 0,
                     "carbohydrates":carbohydrates if carbohydrates is not None else 0}

           density = self.get_density(food_label)
           result["calories_per_ml"] = (result["calories"]/100)*density
           self.cache[key] = result
           self._save_cache_entry(food_label, result)
           return result
        else:
            return None
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