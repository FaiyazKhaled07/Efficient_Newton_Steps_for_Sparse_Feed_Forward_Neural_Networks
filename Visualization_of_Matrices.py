#Visualization of weight matrices
weights = model.layers[7].get_weights()[0]
#print(weights)
plt.figure(figsize=(20,20))
#sns.heatmap(weights, cmap="YlGnBu", annot=True, fmt='.2f', annot_kws={"fontsize":5})
plt.title('Weight Matrix')
#plt.show()

#Visualization of Jacobian matrices
plt.figure(figsize=(35, 35))
sns.heatmap(df_jac_8, cmap='coolwarm', annot=True, fmt='.2f', annot_kws={"size": 10})
plt.title('Tridiagonal Jacobian Matrix of ____ Layer')
plt.xlabel('Input Features')
plt.ylabel('Output Units')
plt.show()
