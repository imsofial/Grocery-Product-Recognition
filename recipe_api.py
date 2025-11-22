import requests
import json

BASE_URL = "https://www.themealdb.com/api/json/v1/1"


def _fetch_json(path: str, params=None):
    """
    Helper to call TheMealDB and return either parsed JSON or an error string.
    """
    url = f"{BASE_URL}/{path}"
    try:
        resp = requests.get(url, params=params, timeout=10)
    except Exception as e:
        return f"Error requesting API: {e}"

    if resp.status_code != 200:
        return f"Error requesting API: {resp.status_code}"

    try:
        return resp.json()
    except Exception as e:
        return f"Error parsing API response: {e}"


def _build_recipes_from_meals(meals, category_filter: str, max_results: int):
    """
    Shared helper: take a list of full meal dicts and turn them into
    our internal recipe format, applying category_filter + max_results.
    Returns: list[dict] (may be empty).
    """
    results = []
    for full in meals:
        if len(results) >= max_results:
            break

        cat = (full.get("strCategory") or "").strip()
        # Apply category filter unless "Any"
        if category_filter != "Any":
            if cat.lower() != category_filter.lower():
                continue

        name = full.get("strMeal", "")
        category = cat
        area = (full.get("strArea") or "").strip()
        image = full.get("strMealThumb", "")
        instructions = full.get("strInstructions", "")

        ingredients = []
        for i in range(1, 21):
            ing = full.get(f"strIngredient{i}")
            measure = full.get(f"strMeasure{i}")
            if ing and ing.strip():
                m = (measure or "").strip()
                ingredients.append(f"{ing.strip()} — {m}")

        recipe = {
            "name": name,
            "category": category,
            "area": area,
            "ingredients": ingredients,
            "instructions": instructions,
            "image": image,
        }
        results.append(recipe)

    return results


def _fallback_search_by_name(ingredient: str, category_filter: str, max_results: int):
    """
    Fallback: search by *meal name* instead of ingredient:
      search.php?s=<ingredient>
    """
    data = _fetch_json("search.php", params={"s": ingredient})
    if isinstance(data, str):
        return data  # error message

    meals = data.get("meals")
    if not meals:
        if category_filter == "Any":
            return f"No recipes found for '{ingredient}'."
        else:
            return f"No {category_filter} recipes found for '{ingredient}'."

    results = _build_recipes_from_meals(meals, category_filter, max_results)
    if not results:
        if category_filter == "Any":
            return f"No recipes found for '{ingredient}'."
        else:
            return f"No {category_filter} recipes found for '{ingredient}'."
    return results


def get_recipes_by_ingredient_raw(
    ingredient: str,
    category_filter: str = "Dessert",
    max_results: int = 5,
):
    """
    Returns recipes as a Python list (no file writing).
    On error or no results, returns a string with a message.

    Main logic:
      1) Try filter.php?i=<ingredient> → meals that CONTAIN this ingredient.
      2) For each meal → lookup.php?i=<idMeal> for full details.
      3) Apply category filter unless category_filter == "Any".
      4) If that yields nothing → fallback to search.php?s=<ingredient>.
    """

    ingredient = ingredient.strip()
    if not ingredient:
        return "Empty ingredient."

    # ---- 1) Try filter by ingredient ----
    data = _fetch_json("filter.php", params={"i": ingredient})
    if isinstance(data, str):
        # error message
        return data

    meals = data.get("meals")

    # If filter.php gives nothing → go straight to name search fallback
    if not meals:
        return _fallback_search_by_name(ingredient, category_filter, max_results)

    # We have some meals with that ingredient, but they are "shallow"
    # (only idMeal, strMeal, etc.). We need full details via lookup.php.
    detailed_meals = []
    for meal in meals:
        meal_id = meal.get("idMeal")
        if not meal_id:
            continue

        detail = _fetch_json("lookup.php", params={"i": meal_id})
        if isinstance(detail, str):
            # error, skip
            continue

        detail_meals = detail.get("meals")
        if not detail_meals:
            continue

        detailed_meals.append(detail_meals[0])

    # Build recipes from detailed meals
    results = _build_recipes_from_meals(detailed_meals, category_filter, max_results)

    # If category filter killed everything or something weird happened -> fallback
    if not results:
        return _fallback_search_by_name(ingredient, category_filter, max_results)

    return results


def get_recipes_by_ingredient(ingredient: str, category_filter="Dessert"):
    """
    Backward-compatible wrapper:
    - calls the raw version
    - saves to recipes.json
    - returns a status string (as in your original code)
    """
    res = get_recipes_by_ingredient_raw(ingredient, category_filter=category_filter)

    if isinstance(res, str):
        # error / not found message
        return res

    with open("recipes.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    return f"Saved {len(res)} recipes to recipes.json"


if __name__ == "__main__":
    ingredient = "apple"
    print(get_recipes_by_ingredient(ingredient, category_filter="Any"))
