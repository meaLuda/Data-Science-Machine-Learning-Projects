import json
import random
from faker import Faker
import openai
fake = Faker()


"""
    I got the following dataset:

    This data includes student lists, academic levels, learning preferences, and interactions with e-learning materials. Also included is a list of test tasks with difficulty and content.

    ```json
    {
    "student_profiles": [
        {
        "id": 1,
        "name": "John Doe",
        "age": 18,
        "gender": "male",
        "academic_ability_level": "high",
        "learning_preferences": ["visual", "interactive"],
        "e_learning_materials": [
            {
            "id": 1,
            "name": "Mathematics 101",
            "difficulty_level": "medium",
            "feedback": "Good job!"
            },
            {
            "id": 2,
            "name": "Science 101",
            "difficulty_level": "high",
            "feedback": "Needs improvement."
            }
        ],
        "test_criteria": [
            {
            "id": 1,
            "question": "What is the formula for calculating the area of a circle?",
            "correct_answer": "πr²",
            "difficulty_level": "medium"
            },
            {
            "id": 2,
            "question": "What is the process of photosynthesis?",
            "correct_answer": "The process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll, releasing oxygen as a byproduct.",
            "difficulty_level": "high"
            }
        ]
        },
    ]
    }

    ```

"""

student_profiles_IRT = [] # IRT = Item Response Theory ~ Dataset
student_profiles_ANN = [] # ANN = Artificial Neural Network ~ Dataset

def generate_student_profiles():
    # we will generate the data we need only ignoring the rest
    

    # test criteria
    accademic_level = ["high", "medium", "low"] # academic levels
    anserered_test_criteria = [True,False] # answered test criteria
    test_criteria = ["medium", "high", "low"] # difficulty levels


    for i in range(100):
        # generate student profiles for IRT
    
        student_profile_IRT = {
            "id": i,


