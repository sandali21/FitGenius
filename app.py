from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the FastAPI app
app = FastAPI()

# CORS middleware for allowing cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the exercise data from an Excel file
exercise_data = pd.read_excel('assets/fitgenius.xlsx')

# Define the schema for the user profile using Pydantic
class UserProfile(BaseModel):
    type: str
    goal: str
    fitness_level: str
    preferences: str

# Step 1: Knowledge-Based Filtering
def filter_exercises(profile: UserProfile, exercise_data: pd.DataFrame) -> pd.DataFrame:
    filtered_exercises = exercise_data[
        (exercise_data['bodyType'].str.lower() == profile.type.lower()) &
        (exercise_data['goal'].str.lower() == profile.goal.lower()) &
        (exercise_data['fitnessLevel'].str.lower() == profile.fitness_level.lower())
    ]
    return filtered_exercises

# Step 2: Cosine Similarity for Refinement
def refine_with_cosine_similarity(profile: UserProfile, exercises: pd.DataFrame) -> pd.DataFrame:
    # Combine user preferences into a single string
    user_pref_str = profile.preferences
    
    # Create a list of exercise descriptions to compare with user preferences
    exercise_descriptions = exercises['description'].tolist()
    
    # Add user preferences as the first entry in the list for similarity comparison
    all_descriptions = [user_pref_str] + exercise_descriptions
    
    # Use TF-IDF Vectorizer to convert text to vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_descriptions)
    
    # Compute cosine similarity between the user preferences and exercises
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    # Add cosine similarity scores to the DataFrame
    exercises['similarity_score'] = cosine_sim.flatten()
    
    # Sort exercises by similarity score in descending order
    sorted_exercises = exercises.sort_values(by='similarity_score', ascending=False)
    
    return sorted_exercises

# API endpoint to generate the workout plan
@app.post("/generate-workout-plan/")
async def generate_workout_plan(profile: UserProfile = Body(...)) -> Dict[str, Any]:
    try:
        # Filter exercises using knowledge-based rules
        filtered_exercises = filter_exercises(profile, exercise_data)
        
        # Refine the workout plan using cosine similarity
        refined_workout_plan = refine_with_cosine_similarity(profile, filtered_exercises)
        
        # Convert the final workout plan to dictionary format for JSON response
        workout_plan = refined_workout_plan.to_dict(orient='records')
        
        return {"workout_plan": workout_plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app using: `uvicorn app:app --reload`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
