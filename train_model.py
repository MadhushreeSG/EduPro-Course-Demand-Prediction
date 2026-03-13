import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Sample dataset
data = pd.DataFrame({
    "CoursePrice":[100,200,150,300,120],
    "CourseDuration":[10,20,15,30,12],
    "CourseRating":[4.5,4.8,4.2,4.9,4.0],
    "InstructorExperience":[5,10,3,15,4],
    "TeacherRating":[4.6,4.9,4.1,4.8,4.2],
    "Enrollment":[200,350,150,500,180]
})

# Features and target
X = data.drop("Enrollment", axis=1)
y = data["Enrollment"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model/demand_model.pkl", "wb"))

print("Model trained and saved!")