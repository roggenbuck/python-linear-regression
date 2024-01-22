import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression

length = 200


# Generate a line with noise
def fuzzy_line(x):
    return x + (random.randint(0, x // 10) ** 1.5)


# Create the dataframe
data = pd.DataFrame(
    {
        "x": list(range(0, length)),
        "y": [fuzzy_line(a) for a in range(0, length)]
    }
)

X = data[["x"]]
y = data[["y"]]

# Run the regression
model = LinearRegression()
model.fit(X, y)

# Plot the data
sns.scatterplot(data, x="x", y="y")
plt.plot(X, model.predict(X), color="blue")

plt.show()
