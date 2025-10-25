import requests
import json

def get_recipes_by_ingredient(ingredient: str, category_filter="Dessert"):
    url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={ingredient}"
    response = requests.get(url)

    if response.status_code != 200:
        return f"Error requesting API: {response.status_code}"

    data = response.json()
    meals = data.get("meals")

    if not meals:
        return f"Recipes with '{ingredient}' not found"

    results = []
    count = 0

    for meal in meals:
        if count >= 5:  # 5 recipes
            break

        if meal["strCategory"].lower() != category_filter.lower():
            continue

        name = meal["strMeal"]
        category = meal["strCategory"]
        area = meal["strArea"]
        image = meal["strMealThumb"]
        instructions = meal["strInstructions"]

        ingredients = []
        for i in range(1, 21):
            ing = meal.get(f"strIngredient{i}")
            measure = meal.get(f"strMeasure{i}")
            if ing and ing.strip():
                ingredients.append(f"{ing.strip()} — {measure.strip()}")

        recipe = {
            "name": name,
            "category": category,
            "area": area,
            "ingredients": ingredients,
            "instructions": instructions,
            "image": image
        }

        results.append(recipe)
        count += 1

    with open("recipes.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return f"Saved {len(results)} recipes to recipes.json"


if __name__ == "__main__":
    ingredient = "apple"
    print(get_recipes_by_ingredient(ingredient))
