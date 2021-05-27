import joblib
pred_model=joblib.load("Salarypredicter-model.pk1")

#Now to predict salary
p=int(input("Enter years of experience-: "))
sal=pred_model.predict([[p]])

#Now print the predicted salary
print("Estimated Salary is: ", round(sal[0],2), "INR.")


