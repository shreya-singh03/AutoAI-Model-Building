import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = pd.read_csv('/content/house_price_dataset.csv')


X = data[['Area']]  
y = data['Pricing']   


model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


print(f"MSE: {mean_squared_error(y, y_pred)}")


area_input = float(input("Enter area in square feet: "))
price_prediction = model.predict([[area_input]])
print(f"Predicted Price: ₹{price_prediction[0]:,.2f}")


plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (₹ in lakhs)")
plt.title("House Price Prediction")
plt.legend()
plt.grid(True)

plt.show()  
