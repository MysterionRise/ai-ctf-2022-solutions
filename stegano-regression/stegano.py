import pickle
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

model = LinearRegression()
loaded_model = pickle.load(open('model', 'rb'))

val = "sssfAfsDfe%%%{dInIisdChdh*e]DHSdbeTNhfhdyeSSWTTFSSSllfjdjs{\\#3fdas34df7adJHHstcsdDFur3sfj_1mdfneypcs0KJDsrsFs7sd4nfec3_sdrufdl35}453"
print(len(val))
res = ""
for pos, i in enumerate(loaded_model.coef_):
    print(i)
    if i == 1:
       res += val[pos]

print(res)
print(len(loaded_model.coef_))
print(loaded_model)