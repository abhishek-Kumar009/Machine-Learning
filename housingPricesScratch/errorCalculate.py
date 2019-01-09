def errorCalc(X,y):
    m = len(X)
    error = 1/(2*m)*sum(pow((X - y),2))
    return error

