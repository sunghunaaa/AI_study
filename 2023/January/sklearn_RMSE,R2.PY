from sklearn.metrics import mean_squared_error,r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))  

print("RMSE : ", RMSE(y_test,y_predict))
  
r2= r2_score(y_test,y_predict)

print("R2 : ", r2)
