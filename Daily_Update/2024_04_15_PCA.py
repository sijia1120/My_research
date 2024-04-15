### ---  Example Codes are from "ML Factor Invesment: Chapter PCA"  --- ###

from sklearn import decomposition 
pca = decomposition.PCA(n_components=7) # we impose the number of components 
pca.fit(training_sample[features_short]) # Performs PCA on smaller number of predictors 
print(pca.explained_variance_ratio_) # Cheking the variance explained per component 
P=pd.DataFrame(pca.components_,columns=features_short).T # Rotation (n x k) = (7 x 7) 
P.columns = ['P' + str(col) for col in P.columns] # tidying up columns names 
P

# ----- Visualize the way the prinicipal components are built ----- 
from pca import pca 
model =pca(n_components =7) # Initialize 
results=model.fit_transform(training_sample[features_short],col_labels=features_short) 
# Fit transform and include the column labels and row labels 
model.biplot(n_feat=7, PC=[0,1],cmap=None, label=None, legend=False) # Make biplot

# ----- Once the rotation is known, it is possible to select a subsample of the transformed data -----
pd.DataFrame( # Using DataFrame format 
  np.matmul( # Matrix product using numpy 
  training_sample[features_short].values,P.values[:, :4]), # Matrix values
  columns=['PC1','PC2','PC3','PC4'] # Change column names
).head()
# Show first 5 lines

# Comments on next steps: ---
# These four factors can then be used as orthogonal features in any ML engine.
# The fact that the features are uncorrelated is undoubtedly an asset.
