

```python
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import string
from pandas import Series, DataFrame
```

# Load Amazon dataset


```python
products = pd.read_csv("D:\ml_data\/amazon_baby.csv")
# products = products.head(100)
```

# Perform text cleaning

## removing punctuation by pandas.str.replace


```python
products['review_clean'] = products['review'].str.replace('[^\w\s]', '')
```

## fill n/a values in the review column with empty strings


```python
products = products.fillna({'review': ''})
```

# Extract Sentiments
## ignore rating = 3 that tends to a neutral sentiment


```python
products = products[products['rating'] != 3]
```

## define positive and negative review


```python
products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)
```

## Split into training and test sets


```python
(train_data,test_data) = train_test_split(products, train_size=0.7, random_state=0)
```


```python
len(test_data)
```




    50026




```python
len(train_data)
```




    116726



# Build the word count vector for each review
use regex token pattern to keep single-letter words


```python
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
```


```python
type(vectorizer)
```




    sklearn.feature_extraction.text.CountVectorizer



First, learn vocabulary from the training data and assign columns to words
Then convert the training data into a sparse matrix


```python
train_matrix = vectorizer.fit_transform(train_data['review_clean'].values.astype('U'))
```


```python
type(train_matrix)
```




    scipy.sparse.csr.csr_matrix




```python
test_matrix = vectorizer.transform(test_data['review_clean'].values.astype('U'))
```

# Train a sentiment classifier with logistic regression


```python
sentiment_model = linear_model.LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
sentiment_model.coef_
```




    array([[-1.36821636e+00,  8.61907787e-04,  1.74107372e-02, ...,
             1.02410276e-02,  2.80385130e-03, -7.17964970e-05]])




```python
type(sentiment_model.coef_)
```




    numpy.ndarray



# Making predictions with logistic regression


```python
sample_test_data = test_data[10:13]
print(sample_test_data)
```

                                                        name  \
    27972             Bumkins Waterproof Superbib, Blue Fizz   
    123316                         Safety 1st High Door Lock   
    60874   DadGear Courier Diaper Bag - Orange Retro Stripe   
    
                                                       review  rating  \
    27972   I love these bibs!  We have about 8 of them in...       5   
    123316  My 4 year old gets up earlier than me, this me...       5   
    60874   love the bag, especially since it's over the s...       4   
    
                                                 review_clean  sentiment  
    27972   I love these bibs  We have about 8 of them in ...          1  
    123316  My 4 year old gets up earlier than me this mea...          1  
    60874   love the bag especially since its over the sho...          1  
    


```python
sample_test_data['review'].index
```




    Int64Index([27972, 123316, 60874], dtype='int64')




```python
# sample_test_data['review'][113598]
# sample_test_data['review'][8]
```


```python
len(sample_test_data['review'])
```




    3




```python
for s in sample_test_data['review']:
    print(s)
    print('___________________________________')
```

    I love these bibs!  We have about 8 of them in different patterns!  They was well!  Even with bleach!We've been using them for about 4 years!
    ___________________________________
    My 4 year old gets up earlier than me, this means if he can be quiet he will sneak downstairs and steal sugar.  I don't usually keep treats (cookies, candies, etc) in the house so he will steal actual sugar.  We tried different disciplines but nothing worked.  He has told us "I don't want to steal the sugar but my belly tells me to."  We tried all sorts of locks, we needed something to go on the door to the downstairs and not his room b/c he needs to be able to leave to use the bathroom.  Now, I can lock the door at night and know that he won't be able to sneak down in the morning and steal sugar, or use knives, or the stove, you know, the typical stuff that 4 year olds try to do when they wake up in the morning (oh, that's not typical you say...hmmmm).  We had tried all the typical door knob locks, and some no drill locks (we were trying not to put holes in as we rent), none of them work, he can defeat them all; they tell me he's gifted...yay for me?  So even though this one requires some drilling it is really a snap to install and as long as you mount it high enough (and there are no available chairs, ladders, precariously stacked boxes available) your child shouldn't be able to get through the door.I should also note, it is easy to open from both sides of the door when locked.  When we first installed it was a little tricky to slide it from the on to off position (red button means it will not automatically lock when you shut the door, green means it will) but it seems it just needed some use to loosen up since it is easy to do now.  I also like the finger jam guard, the 4 year old has a tendency to slam doors, which often has disastrous results when his 2 year old brother acts like his shadow.
    ___________________________________
    love the bag, especially since it's over the shoulder.  i've seen too many peeps with regular shoulder bags sliding down their arm while trying to hold a car seat or baby.  it has a lot of compartments to keep stuff organized and still have enough space in the big pocket for formula and diapers.i have to stop using the wipe holder tho.  it has always been so hard to open, i took skin off my finger today trying to open it.  and it's very hard to fill the wipe holder too.  i haven't found a way to actually remove the holder, so i'll probably just take the back off (so there's not a giant hole in my bag) and use a regular wipe holder inside the bag.
    ___________________________________
    


```python
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'].values.astype('U'))
scores_1 = sentiment_model.decision_function(sample_test_matrix)
print(scores_1)
```

    [ 5.67446022 15.91839528  3.78700706]
    


```python
sample_test_matrix.shape
```




    (3, 110870)




```python
scores_2 = sentiment_model.predict(sample_test_matrix)
print(scores_2)
```

    [1 1 1]
    

# Probability Predictions


```python
import math
```


```python
math.e
```




    2.718281828459045




```python
def probability(score):
    return 1/(1 + math.e**(-score))
```


```python
for i in scores_1:
    print(probability(i))
```

    0.9965792205996046
    0.9999998778963308
    0.977838913549031
    

# Find the most positive (and negative) review
We now turn to examining the full test dataset, test_data, and use sklearn.linear_model.LogisticRegression to form predictions on all of the test data points.


```python
proba = sentiment_model.predict_proba(sample_test_matrix)
```


```python
proba
```




    array([[3.42077940e-03, 9.96579221e-01],
           [1.22103669e-07, 9.99999878e-01],
           [2.21610865e-02, 9.77838914e-01]])



Using the sentiment_model, find the 20 reviews in the entire test_data with the highest probability of being classified as a positive review. We refer to these as the "most positive reviews."


```python
proba_all = sentiment_model.predict_proba(test_matrix)
# test_data['probability'] = sentiment_model.predict(test_data, output_type='probability')
```


```python
# len(proba_all)
type(proba_all)
```




    numpy.ndarray




```python
proba_all
```




    array([[1.64768828e-02, 9.83523117e-01],
           [2.13707041e-01, 7.86292959e-01],
           [1.10050642e-01, 8.89949358e-01],
           ...,
           [9.10265794e-01, 8.97342059e-02],
           [9.11668969e-02, 9.08833103e-01],
           [2.38539982e-06, 9.99997615e-01]])




```python
proba_all_positive = proba_all[:,1]
```


```python
len(proba_all_positive)
```




    50026




```python
type(proba_all_positive)
```




    numpy.ndarray




```python
type(test_data)
```




    pandas.core.frame.DataFrame




```python
test_data.loc[:,'probability'] = proba_all_positive
```

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\indexing.py:357: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[key] = _infer_fill_value(value)
    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\indexing.py:537: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s
    


```python
np.sort(proba_all,axis=0)
```




    array([[0.00000000e+00, 5.28769314e-19],
           [0.00000000e+00, 8.76249710e-16],
           [0.00000000e+00, 4.69016318e-15],
           ...,
           [1.00000000e+00, 1.00000000e+00],
           [1.00000000e+00, 1.00000000e+00],
           [1.00000000e+00, 1.00000000e+00]])




```python
test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>review_clean</th>
      <th>sentiment</th>
      <th>probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>103733</th>
      <td>Luvable Friends Applique Side Closure Bib, Pink</td>
      <td>Got this for a friend for her baby shower. Lov...</td>
      <td>5</td>
      <td>Got this for a friend for her baby shower Love...</td>
      <td>1</td>
      <td>9.835231e-01</td>
    </tr>
    <tr>
      <th>157975</th>
      <td>Infant Optics DXR5 Wall Socket Power Adapter (...</td>
      <td>Ok, so obviously there isn't much to say about...</td>
      <td>5</td>
      <td>Ok so obviously there isnt much to say about a...</td>
      <td>1</td>
      <td>7.862930e-01</td>
    </tr>
    <tr>
      <th>127693</th>
      <td>NUK Active Silicone Spout Learning Cup, Ladybu...</td>
      <td>We have a couple versions of Tommie Tippie and...</td>
      <td>5</td>
      <td>We have a couple versions of Tommie Tippie and...</td>
      <td>1</td>
      <td>8.899494e-01</td>
    </tr>
    <tr>
      <th>113718</th>
      <td>Luvable Friends 12 Pack Washcloths, Blue</td>
      <td>Ordered washcloths pictured but others were se...</td>
      <td>1</td>
      <td>Ordered washcloths pictured but others were se...</td>
      <td>-1</td>
      <td>3.017619e-01</td>
    </tr>
    <tr>
      <th>20416</th>
      <td>Infantino Cart Safari w/Microban</td>
      <td>perfect for what its supposed to be. my 9 mont...</td>
      <td>5</td>
      <td>perfect for what its supposed to be my 9 month...</td>
      <td>1</td>
      <td>9.995342e-01</td>
    </tr>
    <tr>
      <th>62236</th>
      <td>Sesame Street Bath Tub Faucet Cover - Elmo</td>
      <td>Thought this would be great as my son loves El...</td>
      <td>1</td>
      <td>Thought this would be great as my son loves El...</td>
      <td>-1</td>
      <td>3.682715e-01</td>
    </tr>
    <tr>
      <th>93741</th>
      <td>The First Years Ignite Stroller</td>
      <td>We purchased this for a trip to Disney. My 4 y...</td>
      <td>5</td>
      <td>We purchased this for a trip to Disney My 4 ye...</td>
      <td>1</td>
      <td>9.991669e-01</td>
    </tr>
    <tr>
      <th>182819</th>
      <td>Bath Letters And Numbers With Bath Toys Organi...</td>
      <td>Exactly what I was looking for. Foam letters f...</td>
      <td>5</td>
      <td>Exactly what I was looking for Foam letters fo...</td>
      <td>1</td>
      <td>9.955896e-01</td>
    </tr>
    <tr>
      <th>108942</th>
      <td>Britax B-Ready Stroller, Black</td>
      <td>We got this stroller during Britax's ride even...</td>
      <td>5</td>
      <td>We got this stroller during Britaxs ride event...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>63171</th>
      <td>Ju-Ju-Be Paci Pod Pacifier Holder, Lilac Lace</td>
      <td>The kids at my house have moved on past binkie...</td>
      <td>5</td>
      <td>The kids at my house have moved on past binkie...</td>
      <td>1</td>
      <td>8.832268e-01</td>
    </tr>
    <tr>
      <th>27972</th>
      <td>Bumkins Waterproof Superbib, Blue Fizz</td>
      <td>I love these bibs!  We have about 8 of them in...</td>
      <td>5</td>
      <td>I love these bibs  We have about 8 of them in ...</td>
      <td>1</td>
      <td>9.965792e-01</td>
    </tr>
    <tr>
      <th>123316</th>
      <td>Safety 1st High Door Lock</td>
      <td>My 4 year old gets up earlier than me, this me...</td>
      <td>5</td>
      <td>My 4 year old gets up earlier than me this mea...</td>
      <td>1</td>
      <td>9.999999e-01</td>
    </tr>
    <tr>
      <th>60874</th>
      <td>DadGear Courier Diaper Bag - Orange Retro Stripe</td>
      <td>love the bag, especially since it's over the s...</td>
      <td>4</td>
      <td>love the bag especially since its over the sho...</td>
      <td>1</td>
      <td>9.778389e-01</td>
    </tr>
    <tr>
      <th>89381</th>
      <td>Prince Lionheart weePOD, Green</td>
      <td>The splash guard on this seat was the major se...</td>
      <td>5</td>
      <td>The splash guard on this seat was the major se...</td>
      <td>1</td>
      <td>9.897353e-01</td>
    </tr>
    <tr>
      <th>146248</th>
      <td>Prince Lionheart Seat Neat, Brown/Tan</td>
      <td>Have used it for about 3 months now and it's a...</td>
      <td>5</td>
      <td>Have used it for about 3 months now and its a ...</td>
      <td>1</td>
      <td>9.998925e-01</td>
    </tr>
    <tr>
      <th>167897</th>
      <td>Wrapsody Breeze Baby Carrier, Freya, Medium/Large</td>
      <td>This wrap is very thin and lightweight, great ...</td>
      <td>5</td>
      <td>This wrap is very thin and lightweight great f...</td>
      <td>1</td>
      <td>9.460426e-01</td>
    </tr>
    <tr>
      <th>129689</th>
      <td>Noah And Friends High Chair Cover</td>
      <td></td>
      <td>5</td>
      <td>NaN</td>
      <td>1</td>
      <td>8.650732e-01</td>
    </tr>
    <tr>
      <th>54150</th>
      <td>Medela Pump in Style Advanced Backpack</td>
      <td>I got this pump as I went back to work with my...</td>
      <td>5</td>
      <td>I got this pump as I went back to work with my...</td>
      <td>1</td>
      <td>8.229963e-01</td>
    </tr>
    <tr>
      <th>8352</th>
      <td>Baby Bjorn Toilet Trainer</td>
      <td>Before I had purchased this toilet trainer, I ...</td>
      <td>5</td>
      <td>Before I had purchased this toilet trainer I w...</td>
      <td>1</td>
      <td>9.997415e-01</td>
    </tr>
    <tr>
      <th>55700</th>
      <td>HABA Blossom Butterfly Mobile</td>
      <td>My son is 6 weeks old, and we bought this when...</td>
      <td>5</td>
      <td>My son is 6 weeks old and we bought this when ...</td>
      <td>1</td>
      <td>9.980966e-01</td>
    </tr>
    <tr>
      <th>83479</th>
      <td>Dreambaby Super Toy Hammock and Toy Chain</td>
      <td>Unfortunately I like in a cement high rise. I ...</td>
      <td>5</td>
      <td>Unfortunately I like in a cement high rise I c...</td>
      <td>1</td>
      <td>6.319987e-01</td>
    </tr>
    <tr>
      <th>93618</th>
      <td>Sunshine Kids Seat Belt Pillow, Grey</td>
      <td>I purchased this after my husband had bypass s...</td>
      <td>5</td>
      <td>I purchased this after my husband had bypass s...</td>
      <td>1</td>
      <td>9.989821e-01</td>
    </tr>
    <tr>
      <th>130762</th>
      <td>Dreambaby Sliding Door and Window Locks</td>
      <td>The product is expensive and we tried to use i...</td>
      <td>2</td>
      <td>The product is expensive and we tried to use i...</td>
      <td>-1</td>
      <td>5.309546e-02</td>
    </tr>
    <tr>
      <th>73991</th>
      <td>Delta Canton 4-in-1 Convertible Crib, Dark Cherry</td>
      <td>Really nice looking, easy to put together.Look...</td>
      <td>5</td>
      <td>Really nice looking easy to put togetherLooks ...</td>
      <td>1</td>
      <td>9.942311e-01</td>
    </tr>
    <tr>
      <th>174021</th>
      <td>Thermos FUNtainer Monsters University Food Jar...</td>
      <td>The thermos is great for keeping food warm. Th...</td>
      <td>4</td>
      <td>The thermos is great for keeping food warm The...</td>
      <td>1</td>
      <td>9.270045e-01</td>
    </tr>
    <tr>
      <th>17830</th>
      <td>Prince Lionheart Multi-Purpose Toy Hammock</td>
      <td>When I first saw the box I was discouraged, I ...</td>
      <td>5</td>
      <td>When I first saw the box I was discouraged I t...</td>
      <td>1</td>
      <td>9.919657e-01</td>
    </tr>
    <tr>
      <th>62700</th>
      <td>Vulli Chan Pie Gnon Natural Rubber Teether - B...</td>
      <td>I got this instead of the Sophie Giraffe for m...</td>
      <td>4</td>
      <td>I got this instead of the Sophie Giraffe for m...</td>
      <td>1</td>
      <td>5.626660e-01</td>
    </tr>
    <tr>
      <th>33106</th>
      <td>Lamaze Play &amp;amp; Grow Mortimer the Moose Take...</td>
      <td>This is my son's favorite toy. He is 6 months ...</td>
      <td>5</td>
      <td>This is my sons favorite toy He is 6 months an...</td>
      <td>1</td>
      <td>9.957130e-01</td>
    </tr>
    <tr>
      <th>156734</th>
      <td>THE TWIN Z PILLOW - BLUE COLOR COVER The only ...</td>
      <td>The few timew i hace used it, my back ends up ...</td>
      <td>2</td>
      <td>The few timew i hace used it my back ends up a...</td>
      <td>-1</td>
      <td>3.840617e-01</td>
    </tr>
    <tr>
      <th>67422</th>
      <td>Fisher-Price Cheer for Me Potty</td>
      <td>I bought this for our third son. When I read t...</td>
      <td>5</td>
      <td>I bought this for our third son When I read th...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>29152</th>
      <td>Elegant Baby 8 Piece Bath Squirties Gift Set i...</td>
      <td>I added these to my daughters wish list thinki...</td>
      <td>5</td>
      <td>I added these to my daughters wish list thinki...</td>
      <td>1</td>
      <td>9.992346e-01</td>
    </tr>
    <tr>
      <th>145868</th>
      <td>The First Years Deluxe Reclining Feeding Seat,...</td>
      <td>We use this for travelling or at family's hous...</td>
      <td>4</td>
      <td>We use this for travelling or at familys house...</td>
      <td>1</td>
      <td>8.718506e-01</td>
    </tr>
    <tr>
      <th>133924</th>
      <td>C.R. Gibson Keepsake Memory Book of Baby's Fir...</td>
      <td>What a great idea to have such a special keeps...</td>
      <td>4</td>
      <td>What a great idea to have such a special keeps...</td>
      <td>1</td>
      <td>8.313210e-01</td>
    </tr>
    <tr>
      <th>144259</th>
      <td>Baby Book in a Box Memory Keepsake Organizer</td>
      <td>Wish I had discovered this when my children we...</td>
      <td>5</td>
      <td>Wish I had discovered this when my children we...</td>
      <td>1</td>
      <td>9.996570e-01</td>
    </tr>
    <tr>
      <th>155251</th>
      <td>VTech Communications Safe &amp;amp; Sound Video Ca...</td>
      <td>I love having a second camera.  It works great...</td>
      <td>5</td>
      <td>I love having a second camera  It works great ...</td>
      <td>1</td>
      <td>9.991212e-01</td>
    </tr>
    <tr>
      <th>49516</th>
      <td>Mary Meyer Christening Lamb Blanket - 14 Inches</td>
      <td>The lamb was exactly as pictured and arrived o...</td>
      <td>4</td>
      <td>The lamb was exactly as pictured and arrived o...</td>
      <td>1</td>
      <td>6.219870e-02</td>
    </tr>
    <tr>
      <th>139738</th>
      <td>Fisher-Price Calming Waters Vibration Bathing Tub</td>
      <td>This is perfect for my newborn! She loves bath...</td>
      <td>5</td>
      <td>This is perfect for my newborn She loves bath ...</td>
      <td>1</td>
      <td>9.999232e-01</td>
    </tr>
    <tr>
      <th>155993</th>
      <td>Similac SimplySmart Bottle, 4 Ounce</td>
      <td>I bought these bottles for my Grand daughter. ...</td>
      <td>5</td>
      <td>I bought these bottles for my Grand daughter T...</td>
      <td>1</td>
      <td>9.979781e-01</td>
    </tr>
    <tr>
      <th>28007</th>
      <td>Badger Basket Three Basket Set, Pink</td>
      <td>I got a pinkmone and it was barely pink. It as...</td>
      <td>4</td>
      <td>I got a pinkmone and it was barely pink It as ...</td>
      <td>1</td>
      <td>3.436229e-01</td>
    </tr>
    <tr>
      <th>124413</th>
      <td>Summer Infant Lil' Luxuries Whirlpool, Bubblin...</td>
      <td>There's not much I could say about this tub th...</td>
      <td>1</td>
      <td>Theres not much I could say about this tub tha...</td>
      <td>-1</td>
      <td>4.901535e-04</td>
    </tr>
    <tr>
      <th>114981</th>
      <td>Fisher-Price Sing with Me Step Stool</td>
      <td>Bought this for my 3 year old grandson to enco...</td>
      <td>5</td>
      <td>Bought this for my 3 year old grandson to enco...</td>
      <td>1</td>
      <td>9.977971e-01</td>
    </tr>
    <tr>
      <th>115743</th>
      <td>Kinderglo Portable Fun and Safe Rechargeable N...</td>
      <td>My daughter was so excited to get this night l...</td>
      <td>1</td>
      <td>My daughter was so excited to get this night l...</td>
      <td>-1</td>
      <td>1.153964e-02</td>
    </tr>
    <tr>
      <th>109348</th>
      <td>WubbaNub Lamb</td>
      <td>This most adorable lamb has made giving a paci...</td>
      <td>5</td>
      <td>This most adorable lamb has made giving a paci...</td>
      <td>1</td>
      <td>9.999833e-01</td>
    </tr>
    <tr>
      <th>73490</th>
      <td>Regalo Easy Open 50 Inch Super Wide Walk Thru ...</td>
      <td>Returned item. With a 10 month old who likes t...</td>
      <td>1</td>
      <td>Returned item With a 10 month old who likes to...</td>
      <td>-1</td>
      <td>2.238597e-01</td>
    </tr>
    <tr>
      <th>10461</th>
      <td>Regalo Easy Diner Portable Hook-On High Chair</td>
      <td>Bought two as baby shower gifts - I used a cha...</td>
      <td>5</td>
      <td>Bought two as baby shower gifts  I used a chai...</td>
      <td>1</td>
      <td>9.870586e-01</td>
    </tr>
    <tr>
      <th>120390</th>
      <td>Munchkin Extending Extra Tall and Wide Metal G...</td>
      <td>Assembly wasn't too terrible - took about an h...</td>
      <td>2</td>
      <td>Assembly wasnt too terrible  took about an hou...</td>
      <td>-1</td>
      <td>2.389540e-10</td>
    </tr>
    <tr>
      <th>56823</th>
      <td>Bunnies by the Bay Wee Plush, Ittybit</td>
      <td>I bought this bunny for my 4 month old grand d...</td>
      <td>5</td>
      <td>I bought this bunny for my 4 month old grand d...</td>
      <td>1</td>
      <td>9.558778e-01</td>
    </tr>
    <tr>
      <th>110189</th>
      <td>eWonderWorld 24&amp;quot; X 24&amp;quot; X~9/16&amp;quot; ...</td>
      <td>Perfect for a little one who is learning to cr...</td>
      <td>4</td>
      <td>Perfect for a little one who is learning to cr...</td>
      <td>1</td>
      <td>9.975703e-01</td>
    </tr>
    <tr>
      <th>151692</th>
      <td>WubbaNub Infant Plush Pacifier - Limited Editi...</td>
      <td>My son has always loved these. So when I seen ...</td>
      <td>5</td>
      <td>My son has always loved these So when I seen t...</td>
      <td>1</td>
      <td>9.487158e-01</td>
    </tr>
    <tr>
      <th>134663</th>
      <td>Infant Optics DXR-5 2.4 GHz Digital Video Baby...</td>
      <td>I purchased this monitor in November 2012, jus...</td>
      <td>4</td>
      <td>I purchased this monitor in November 2012 just...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>14881</th>
      <td>Magic Moments Learning Seat</td>
      <td>We purchased this bouncer for our 2.5 month ol...</td>
      <td>5</td>
      <td>We purchased this bouncer for our 25 month old...</td>
      <td>1</td>
      <td>9.995947e-01</td>
    </tr>
    <tr>
      <th>106787</th>
      <td>Wubbanub Infant Pacifier ~ Giraffe &amp;amp; Lion</td>
      <td>My boys love Wubba Nubba.  I'm almost sad to s...</td>
      <td>5</td>
      <td>My boys love Wubba Nubba  Im almost sad to say...</td>
      <td>1</td>
      <td>8.969469e-01</td>
    </tr>
    <tr>
      <th>9918</th>
      <td>Cosco Alpha Omega Elite Convertible Car Seat</td>
      <td>This car seat does have a bad smell.  I took a...</td>
      <td>4</td>
      <td>This car seat does have a bad smell  I took al...</td>
      <td>1</td>
      <td>9.993759e-01</td>
    </tr>
    <tr>
      <th>79920</th>
      <td>Simple Wishes Hands-Free Breastpump Bra, Pink,...</td>
      <td>I ordered this product after reading that the ...</td>
      <td>5</td>
      <td>I ordered this product after reading that the ...</td>
      <td>1</td>
      <td>9.647574e-01</td>
    </tr>
    <tr>
      <th>31318</th>
      <td>Prince Lionheart Wheely Bug, Ladybug, Large</td>
      <td>This is the best ride-on toy I've seen.  The w...</td>
      <td>5</td>
      <td>This is the best rideon toy Ive seen  The whee...</td>
      <td>1</td>
      <td>9.950470e-01</td>
    </tr>
    <tr>
      <th>54722</th>
      <td>Graco Nautilus 3-in-1 Car Seat, Matrix</td>
      <td>I bought this car seat for my grand daughter, ...</td>
      <td>2</td>
      <td>I bought this car seat for my grand daughter w...</td>
      <td>-1</td>
      <td>9.279842e-01</td>
    </tr>
    <tr>
      <th>106191</th>
      <td>Lamaze High-Contrast Flip-Flop Bug Rattle</td>
      <td>One of her favorite toys. Excellent soft rattl...</td>
      <td>5</td>
      <td>One of her favorite toys Excellent soft rattle...</td>
      <td>1</td>
      <td>9.997943e-01</td>
    </tr>
    <tr>
      <th>79222</th>
      <td>Britax Chaperone Stroller, Red Mill</td>
      <td>way too expensive i really wanted this travel ...</td>
      <td>1</td>
      <td>way too expensive i really wanted this travel ...</td>
      <td>-1</td>
      <td>8.973421e-02</td>
    </tr>
    <tr>
      <th>93173</th>
      <td>Contours Options Tandem II Stroller, Tangerine</td>
      <td>The only reason I am writing this review is be...</td>
      <td>1</td>
      <td>The only reason I am writing this review is be...</td>
      <td>-1</td>
      <td>9.088331e-01</td>
    </tr>
    <tr>
      <th>86465</th>
      <td>Fisher-Price Newborn Rock 'n Play Sleeper, Yellow</td>
      <td>Update: My son is now 13 months old.  He slept...</td>
      <td>5</td>
      <td>Update My son is now 13 months old  He slept i...</td>
      <td>1</td>
      <td>9.999976e-01</td>
    </tr>
  </tbody>
</table>
<p>50026 rows × 6 columns</p>
</div>




```python
type(test_data)
```




    pandas.core.frame.DataFrame




```python
test_data.sort_values(by='probability', ascending=False, na_position='first')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>review_clean</th>
      <th>sentiment</th>
      <th>probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>147975</th>
      <td>Baby Jogger City Mini GT Single Stroller, Shad...</td>
      <td>Let me start by saying that I have gone throug...</td>
      <td>5</td>
      <td>Let me start by saying that I have gone throug...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>167047</th>
      <td>Inglesina 2013 Trip Stroller, Lampone Purple</td>
      <td>I did many hours of research reading reviews a...</td>
      <td>5</td>
      <td>I did many hours of research reading reviews a...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>134265</th>
      <td>UPPAbaby Cruz Stroller, Denny</td>
      <td>We bought this stroller after selling our belo...</td>
      <td>5</td>
      <td>We bought this stroller after selling our belo...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>106455</th>
      <td>Quinny Senzz 2011 Fashion Stroller, Star</td>
      <td>I am very pleased overall with the Quinny Senz...</td>
      <td>4</td>
      <td>I am very pleased overall with the Quinny Senz...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>171041</th>
      <td>CTA Digital 2-in-1 iPotty with Activity Seat f...</td>
      <td>I'll just say in advance that the haters can j...</td>
      <td>5</td>
      <td>Ill just say in advance that the haters can ju...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>76549</th>
      <td>Britax Advocate 65 CS Click &amp;amp; Safe Convert...</td>
      <td>The Britax Advocate CS appears similar to the ...</td>
      <td>4</td>
      <td>The Britax Advocate CS appears similar to the ...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>59320</th>
      <td>Evenflo Aura Select Travel System - Caroline</td>
      <td>I'm about to be a first-time mom, so I spent w...</td>
      <td>5</td>
      <td>Im about to be a firsttime mom so I spent week...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>123632</th>
      <td>Zooper 2011 Waltz Standard Stroller, Flax Brown</td>
      <td>I did a TON of research before I purchased thi...</td>
      <td>5</td>
      <td>I did a TON of research before I purchased thi...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>41763</th>
      <td>Kolcraft Contours Lite Stroller Plus with iPod...</td>
      <td>After considering several lightweight stroller...</td>
      <td>4</td>
      <td>After considering several lightweight stroller...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>48158</th>
      <td>Phil &amp;amp; Ted's &amp;quot;2008 Version 2&amp;quot; Sp...</td>
      <td>We're keeping this stroller! After much resear...</td>
      <td>4</td>
      <td>Were keeping this stroller After much research...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>88680</th>
      <td>ERGObaby Original Baby Carrier, Galaxy Grey</td>
      <td>After reading many online reviews, asking othe...</td>
      <td>5</td>
      <td>After reading many online reviews asking other...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>21557</th>
      <td>Joovy Caboose Stand On Tandem Stroller, Black</td>
      <td>Ok, I read all the reviews already posted here...</td>
      <td>5</td>
      <td>Ok I read all the reviews already posted here ...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>93690</th>
      <td>The First Years Ignite Stroller</td>
      <td>The last thing we wanted was to purchase more ...</td>
      <td>5</td>
      <td>The last thing we wanted was to purchase more ...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>108803</th>
      <td>Chicco Keyfit 22 Pound Infant Car Seat And Bas...</td>
      <td>I bought this right before the KeyFit 30 came ...</td>
      <td>5</td>
      <td>I bought this right before the KeyFit 30 came ...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>88659</th>
      <td>ERGObaby Original Baby Carrier, Galaxy Grey</td>
      <td>We purchased this carrier after a recommendati...</td>
      <td>5</td>
      <td>We purchased this carrier after a recommendati...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>166929</th>
      <td>Britax Pavilion 70-G3 Convertible Car Seat Sea...</td>
      <td>We LOVE this seat! As parents to 8 children ra...</td>
      <td>5</td>
      <td>We LOVE this seat As parents to 8 children ran...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>103186</th>
      <td>Thirsties Duo Wrap Snap, Ocean Blue, Size One ...</td>
      <td>I am reviewing these covers because reading re...</td>
      <td>5</td>
      <td>I am reviewing these covers because reading re...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>152013</th>
      <td>UPPAbaby Vista Stroller, Denny</td>
      <td>I researched strollers for months and months b...</td>
      <td>5</td>
      <td>I researched strollers for months and months b...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>158209</th>
      <td>Ubbi Cloth Diaper Pail Liner</td>
      <td>(updated 3.22.13) After extensive research, tr...</td>
      <td>5</td>
      <td>updated 32213 After extensive research trial a...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>50735</th>
      <td>Joovy Zoom 360 Swivel Wheel Jogging Stroller, ...</td>
      <td>The joovy zoom 360 was the perfect solution fo...</td>
      <td>5</td>
      <td>The joovy zoom 360 was the perfect solution fo...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>140780</th>
      <td>Diono RadianR100 Convertible Car Seat, Dune</td>
      <td>i bought this when the seat was owned by Sunsh...</td>
      <td>5</td>
      <td>i bought this when the seat was owned by Sunsh...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>108943</th>
      <td>Britax B-Ready Stroller, Black</td>
      <td>Some differences with Uppababy Vs. Britax B-Re...</td>
      <td>4</td>
      <td>Some differences with Uppababy Vs Britax BRead...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>111746</th>
      <td>Baby Jogger 2011 City Mini Double Stroller, Bl...</td>
      <td>Before purchasing this stroller, I read severa...</td>
      <td>5</td>
      <td>Before purchasing this stroller I read several...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>135152</th>
      <td>Maxi-Cosi Pria 70 with Tiny Fit Convertible Ca...</td>
      <td>We've been using Britax for our boy (now 14 mo...</td>
      <td>5</td>
      <td>Weve been using Britax for our boy now 14 mont...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>129722</th>
      <td>Bumbleride 2011 Flite Lightweight Compact Trav...</td>
      <td>This is a review of the 2012 Bumbleride Flite ...</td>
      <td>5</td>
      <td>This is a review of the 2012 Bumbleride Flite ...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>100166</th>
      <td>Infantino Wrap and Tie Baby Carrier, Black Blu...</td>
      <td>I bought this carrier when my daughter was abo...</td>
      <td>5</td>
      <td>I bought this carrier when my daughter was abo...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>129212</th>
      <td>Baby Jogger 2011 City Select Stroller in Ameth...</td>
      <td>I have the Baby Jogger City Select with Second...</td>
      <td>5</td>
      <td>I have the Baby Jogger City Select with Second...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>14008</th>
      <td>Stork Craft Beatrice Combo Tower Chest, White</td>
      <td>I bought the tower despite the bad reviews and...</td>
      <td>5</td>
      <td>I bought the tower despite the bad reviews and...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>162687</th>
      <td>Joovy Caboose Too Rear Seat, Greenie</td>
      <td>We are thrilled with this rear seat. This litt...</td>
      <td>5</td>
      <td>We are thrilled with this rear seat This littl...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>166827</th>
      <td>Britax Boulevard 70-G3 Convertible Car Seat Se...</td>
      <td>We just purchased this seat for our eight mont...</td>
      <td>5</td>
      <td>We just purchased this seat for our eight mont...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4835</th>
      <td>JJ Cole Premaxx Sling Carrier - New Edition Re...</td>
      <td>I bought this for my daughter while I was stil...</td>
      <td>1</td>
      <td>I bought this for my daughter while I was stil...</td>
      <td>-1</td>
      <td>1.595787e-09</td>
    </tr>
    <tr>
      <th>3746</th>
      <td>Playtex Diaper Genie - First Refill Included</td>
      <td>Prior to parenthood, I had heard several paren...</td>
      <td>1</td>
      <td>Prior to parenthood I had heard several parent...</td>
      <td>-1</td>
      <td>1.505042e-09</td>
    </tr>
    <tr>
      <th>13712</th>
      <td>Badger Basket Elegance Round Baby Bassinet, Wh...</td>
      <td>I registered for this item &amp; it was ordered fo...</td>
      <td>1</td>
      <td>I registered for this item  it was ordered for...</td>
      <td>-1</td>
      <td>1.412743e-09</td>
    </tr>
    <tr>
      <th>5033</th>
      <td>Playtex 3 Pack BPA Free VentAire Wide Bottles,...</td>
      <td>Initially, I thought these angled bottles make...</td>
      <td>1</td>
      <td>Initially I thought these angled bottles make ...</td>
      <td>-1</td>
      <td>1.154985e-09</td>
    </tr>
    <tr>
      <th>7075</th>
      <td>Peace of Mind Two 900 Mhz Baby Receivers, Monitor</td>
      <td>If we only knew when we registered how terribl...</td>
      <td>1</td>
      <td>If we only knew when we registered how terribl...</td>
      <td>-1</td>
      <td>1.019131e-09</td>
    </tr>
    <tr>
      <th>115108</th>
      <td>Blueberry Deluxe Diaper, Cow</td>
      <td>I purchased the Blueberry One Sized Bamboo Del...</td>
      <td>2</td>
      <td>I purchased the Blueberry One Sized Bamboo Del...</td>
      <td>-1</td>
      <td>9.164181e-10</td>
    </tr>
    <tr>
      <th>70137</th>
      <td>Lulu Ladybug Rocker by Rockabye</td>
      <td>Here's my letter I sent to Rockabye:I purchase...</td>
      <td>1</td>
      <td>Heres my letter I sent to RockabyeI purchased ...</td>
      <td>-1</td>
      <td>4.367577e-10</td>
    </tr>
    <tr>
      <th>72262</th>
      <td>Baby Trend High Chair, Chickadee</td>
      <td>We recently moved from Okinawa back to America...</td>
      <td>2</td>
      <td>We recently moved from Okinawa back to America...</td>
      <td>-1</td>
      <td>4.106157e-10</td>
    </tr>
    <tr>
      <th>124145</th>
      <td>Philips AVENT BPA Free Twin Electric Breast Pump</td>
      <td>UPDATED REVIEW:So, after 2 month of use (once ...</td>
      <td>1</td>
      <td>UPDATED REVIEWSo after 2 month of use once a d...</td>
      <td>-1</td>
      <td>3.359271e-10</td>
    </tr>
    <tr>
      <th>41581</th>
      <td>Newborn Baby Pea in The Pod Halloween Costume,...</td>
      <td>Looks really cute, however, the cloth smells f...</td>
      <td>1</td>
      <td>Looks really cute however the cloth smells fun...</td>
      <td>-1</td>
      <td>3.141029e-10</td>
    </tr>
    <tr>
      <th>67615</th>
      <td>Fisher-Price Butterfly Garden Cradle 'n Swing ...</td>
      <td>My 2nd daughter is 10 weeks old and a little c...</td>
      <td>1</td>
      <td>My 2nd daughter is 10 weeks old and a little c...</td>
      <td>-1</td>
      <td>2.780037e-10</td>
    </tr>
    <tr>
      <th>120390</th>
      <td>Munchkin Extending Extra Tall and Wide Metal G...</td>
      <td>Assembly wasn't too terrible - took about an h...</td>
      <td>2</td>
      <td>Assembly wasnt too terrible  took about an hou...</td>
      <td>-1</td>
      <td>2.389540e-10</td>
    </tr>
    <tr>
      <th>107148</th>
      <td>Tadpoles 36 Sq Ft ABC Floor Mat, Pink/Brown</td>
      <td>I have read the reviews after I bought these (...</td>
      <td>4</td>
      <td>I have read the reviews after I bought these r...</td>
      <td>1</td>
      <td>1.938740e-10</td>
    </tr>
    <tr>
      <th>54418</th>
      <td>Fisher-Price Zen Collection Gliding Bassinet</td>
      <td>THIS BASSINET IS OVERPRICED AND RIDICULOUS.  I...</td>
      <td>2</td>
      <td>THIS BASSINET IS OVERPRICED AND RIDICULOUS  If...</td>
      <td>-1</td>
      <td>1.801620e-10</td>
    </tr>
    <tr>
      <th>75995</th>
      <td>Peg-Perego Tatamia High Chair, White Latte</td>
      <td>Edited to Add 6/4/2010:  Just wanted to add th...</td>
      <td>1</td>
      <td>Edited to Add 642010  Just wanted to add that ...</td>
      <td>-1</td>
      <td>5.132764e-11</td>
    </tr>
    <tr>
      <th>31741</th>
      <td>Regalo My Cot Portable Bed, Royal Blue</td>
      <td>If I could give this product zero stars I woul...</td>
      <td>1</td>
      <td>If I could give this product zero stars I woul...</td>
      <td>-1</td>
      <td>4.603205e-11</td>
    </tr>
    <tr>
      <th>78811</th>
      <td>Tike Tech Single City X4 Swivel Stroller, Paci...</td>
      <td>I initially thought this was a very sturdy, lo...</td>
      <td>2</td>
      <td>I initially thought this was a very sturdy lon...</td>
      <td>-1</td>
      <td>3.269938e-11</td>
    </tr>
    <tr>
      <th>73851</th>
      <td>Stork Craft Rochester Stages Crib with Drawer</td>
      <td>I ordered this crib because I really liked the...</td>
      <td>2</td>
      <td>I ordered this crib because I really liked the...</td>
      <td>-1</td>
      <td>3.203031e-11</td>
    </tr>
    <tr>
      <th>145109</th>
      <td>Withings Smart Baby Monitor, White</td>
      <td>Ok, the good.  During the day, the quality of ...</td>
      <td>1</td>
      <td>Ok the good  During the day the quality of the...</td>
      <td>-1</td>
      <td>2.419640e-11</td>
    </tr>
    <tr>
      <th>179566</th>
      <td>Summer Infant Dual Coverage Digital Color Vide...</td>
      <td>This was possibly the most disappointing elect...</td>
      <td>2</td>
      <td>This was possibly the most disappointing elect...</td>
      <td>-1</td>
      <td>9.570229e-12</td>
    </tr>
    <tr>
      <th>38270</th>
      <td>Nuby No Spill Flip-it Cup, 12 Ounce, Colors Ma...</td>
      <td>I had a frustrating experience with these two ...</td>
      <td>4</td>
      <td>I had a frustrating experience with these two ...</td>
      <td>1</td>
      <td>5.553832e-12</td>
    </tr>
    <tr>
      <th>20331</th>
      <td>Cabinet Flex-Lock (2 pack) from Safety First</td>
      <td>I am a mother of 4 children, ages 6-18 months....</td>
      <td>5</td>
      <td>I am a mother of 4 children ages 618 months  I...</td>
      <td>1</td>
      <td>4.182831e-13</td>
    </tr>
    <tr>
      <th>57234</th>
      <td>Dream On Me Bassinet, Blue</td>
      <td>My husband and I are VERY disappointed and sho...</td>
      <td>1</td>
      <td>My husband and I are VERY disappointed and sho...</td>
      <td>-1</td>
      <td>2.279526e-13</td>
    </tr>
    <tr>
      <th>48694</th>
      <td>Adiri BPA Free Natural Nurser Ultimate Bottle ...</td>
      <td>I will try to write an objective review of the...</td>
      <td>2</td>
      <td>I will try to write an objective review of the...</td>
      <td>-1</td>
      <td>1.520506e-13</td>
    </tr>
    <tr>
      <th>92570</th>
      <td>Munchkin Arm and Hammer Diaper Pail, White</td>
      <td>I would recommend in the strongest possible wa...</td>
      <td>1</td>
      <td>I would recommend in the strongest possible wa...</td>
      <td>-1</td>
      <td>7.840347e-14</td>
    </tr>
    <tr>
      <th>99594</th>
      <td>Valco Baby Tri-mode Twin Stroller EX- Hot Choc...</td>
      <td>I give one star to the dimension: 1. being 29 ...</td>
      <td>1</td>
      <td>I give one star to the dimension 1 being 29 in...</td>
      <td>-1</td>
      <td>6.592510e-14</td>
    </tr>
    <tr>
      <th>131738</th>
      <td>Kids Line Cascade Bow Diaper Bag, Black</td>
      <td>I purchased this in the black color.  For some...</td>
      <td>2</td>
      <td>I purchased this in the black color  For some ...</td>
      <td>-1</td>
      <td>1.286862e-14</td>
    </tr>
    <tr>
      <th>10370</th>
      <td>Wimmer-Ferguson Infant Stim-Mobile</td>
      <td>This product should be in the hall of fame sol...</td>
      <td>1</td>
      <td>This product should be in the hall of fame sol...</td>
      <td>-1</td>
      <td>4.690163e-15</td>
    </tr>
    <tr>
      <th>89902</th>
      <td>Peg-Perego Aria Twin Stroller, Java</td>
      <td>I am so incredibly disappointed with the strol...</td>
      <td>1</td>
      <td>I am so incredibly disappointed with the strol...</td>
      <td>-1</td>
      <td>8.762497e-16</td>
    </tr>
    <tr>
      <th>10180</th>
      <td>Arms Reach Co-Sleeper brand Mini Co-Sleeper Ba...</td>
      <td>Please see my email to the company:Hello,I am ...</td>
      <td>1</td>
      <td>Please see my email to the companyHelloI am wr...</td>
      <td>-1</td>
      <td>5.287693e-19</td>
    </tr>
  </tbody>
</table>
<p>50026 rows × 6 columns</p>
</div>




```python
test_data.sort_values(by='probability', ascending=True, na_position='first')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>review_clean</th>
      <th>sentiment</th>
      <th>probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10180</th>
      <td>Arms Reach Co-Sleeper brand Mini Co-Sleeper Ba...</td>
      <td>Please see my email to the company:Hello,I am ...</td>
      <td>1</td>
      <td>Please see my email to the companyHelloI am wr...</td>
      <td>-1</td>
      <td>5.287693e-19</td>
    </tr>
    <tr>
      <th>89902</th>
      <td>Peg-Perego Aria Twin Stroller, Java</td>
      <td>I am so incredibly disappointed with the strol...</td>
      <td>1</td>
      <td>I am so incredibly disappointed with the strol...</td>
      <td>-1</td>
      <td>8.762497e-16</td>
    </tr>
    <tr>
      <th>10370</th>
      <td>Wimmer-Ferguson Infant Stim-Mobile</td>
      <td>This product should be in the hall of fame sol...</td>
      <td>1</td>
      <td>This product should be in the hall of fame sol...</td>
      <td>-1</td>
      <td>4.690163e-15</td>
    </tr>
    <tr>
      <th>131738</th>
      <td>Kids Line Cascade Bow Diaper Bag, Black</td>
      <td>I purchased this in the black color.  For some...</td>
      <td>2</td>
      <td>I purchased this in the black color  For some ...</td>
      <td>-1</td>
      <td>1.286862e-14</td>
    </tr>
    <tr>
      <th>99594</th>
      <td>Valco Baby Tri-mode Twin Stroller EX- Hot Choc...</td>
      <td>I give one star to the dimension: 1. being 29 ...</td>
      <td>1</td>
      <td>I give one star to the dimension 1 being 29 in...</td>
      <td>-1</td>
      <td>6.592510e-14</td>
    </tr>
    <tr>
      <th>92570</th>
      <td>Munchkin Arm and Hammer Diaper Pail, White</td>
      <td>I would recommend in the strongest possible wa...</td>
      <td>1</td>
      <td>I would recommend in the strongest possible wa...</td>
      <td>-1</td>
      <td>7.840347e-14</td>
    </tr>
    <tr>
      <th>48694</th>
      <td>Adiri BPA Free Natural Nurser Ultimate Bottle ...</td>
      <td>I will try to write an objective review of the...</td>
      <td>2</td>
      <td>I will try to write an objective review of the...</td>
      <td>-1</td>
      <td>1.520506e-13</td>
    </tr>
    <tr>
      <th>57234</th>
      <td>Dream On Me Bassinet, Blue</td>
      <td>My husband and I are VERY disappointed and sho...</td>
      <td>1</td>
      <td>My husband and I are VERY disappointed and sho...</td>
      <td>-1</td>
      <td>2.279526e-13</td>
    </tr>
    <tr>
      <th>20331</th>
      <td>Cabinet Flex-Lock (2 pack) from Safety First</td>
      <td>I am a mother of 4 children, ages 6-18 months....</td>
      <td>5</td>
      <td>I am a mother of 4 children ages 618 months  I...</td>
      <td>1</td>
      <td>4.182831e-13</td>
    </tr>
    <tr>
      <th>38270</th>
      <td>Nuby No Spill Flip-it Cup, 12 Ounce, Colors Ma...</td>
      <td>I had a frustrating experience with these two ...</td>
      <td>4</td>
      <td>I had a frustrating experience with these two ...</td>
      <td>1</td>
      <td>5.553832e-12</td>
    </tr>
    <tr>
      <th>179566</th>
      <td>Summer Infant Dual Coverage Digital Color Vide...</td>
      <td>This was possibly the most disappointing elect...</td>
      <td>2</td>
      <td>This was possibly the most disappointing elect...</td>
      <td>-1</td>
      <td>9.570229e-12</td>
    </tr>
    <tr>
      <th>145109</th>
      <td>Withings Smart Baby Monitor, White</td>
      <td>Ok, the good.  During the day, the quality of ...</td>
      <td>1</td>
      <td>Ok the good  During the day the quality of the...</td>
      <td>-1</td>
      <td>2.419640e-11</td>
    </tr>
    <tr>
      <th>73851</th>
      <td>Stork Craft Rochester Stages Crib with Drawer</td>
      <td>I ordered this crib because I really liked the...</td>
      <td>2</td>
      <td>I ordered this crib because I really liked the...</td>
      <td>-1</td>
      <td>3.203031e-11</td>
    </tr>
    <tr>
      <th>78811</th>
      <td>Tike Tech Single City X4 Swivel Stroller, Paci...</td>
      <td>I initially thought this was a very sturdy, lo...</td>
      <td>2</td>
      <td>I initially thought this was a very sturdy lon...</td>
      <td>-1</td>
      <td>3.269938e-11</td>
    </tr>
    <tr>
      <th>31741</th>
      <td>Regalo My Cot Portable Bed, Royal Blue</td>
      <td>If I could give this product zero stars I woul...</td>
      <td>1</td>
      <td>If I could give this product zero stars I woul...</td>
      <td>-1</td>
      <td>4.603205e-11</td>
    </tr>
    <tr>
      <th>75995</th>
      <td>Peg-Perego Tatamia High Chair, White Latte</td>
      <td>Edited to Add 6/4/2010:  Just wanted to add th...</td>
      <td>1</td>
      <td>Edited to Add 642010  Just wanted to add that ...</td>
      <td>-1</td>
      <td>5.132764e-11</td>
    </tr>
    <tr>
      <th>54418</th>
      <td>Fisher-Price Zen Collection Gliding Bassinet</td>
      <td>THIS BASSINET IS OVERPRICED AND RIDICULOUS.  I...</td>
      <td>2</td>
      <td>THIS BASSINET IS OVERPRICED AND RIDICULOUS  If...</td>
      <td>-1</td>
      <td>1.801620e-10</td>
    </tr>
    <tr>
      <th>107148</th>
      <td>Tadpoles 36 Sq Ft ABC Floor Mat, Pink/Brown</td>
      <td>I have read the reviews after I bought these (...</td>
      <td>4</td>
      <td>I have read the reviews after I bought these r...</td>
      <td>1</td>
      <td>1.938740e-10</td>
    </tr>
    <tr>
      <th>120390</th>
      <td>Munchkin Extending Extra Tall and Wide Metal G...</td>
      <td>Assembly wasn't too terrible - took about an h...</td>
      <td>2</td>
      <td>Assembly wasnt too terrible  took about an hou...</td>
      <td>-1</td>
      <td>2.389540e-10</td>
    </tr>
    <tr>
      <th>67615</th>
      <td>Fisher-Price Butterfly Garden Cradle 'n Swing ...</td>
      <td>My 2nd daughter is 10 weeks old and a little c...</td>
      <td>1</td>
      <td>My 2nd daughter is 10 weeks old and a little c...</td>
      <td>-1</td>
      <td>2.780037e-10</td>
    </tr>
    <tr>
      <th>41581</th>
      <td>Newborn Baby Pea in The Pod Halloween Costume,...</td>
      <td>Looks really cute, however, the cloth smells f...</td>
      <td>1</td>
      <td>Looks really cute however the cloth smells fun...</td>
      <td>-1</td>
      <td>3.141029e-10</td>
    </tr>
    <tr>
      <th>124145</th>
      <td>Philips AVENT BPA Free Twin Electric Breast Pump</td>
      <td>UPDATED REVIEW:So, after 2 month of use (once ...</td>
      <td>1</td>
      <td>UPDATED REVIEWSo after 2 month of use once a d...</td>
      <td>-1</td>
      <td>3.359271e-10</td>
    </tr>
    <tr>
      <th>72262</th>
      <td>Baby Trend High Chair, Chickadee</td>
      <td>We recently moved from Okinawa back to America...</td>
      <td>2</td>
      <td>We recently moved from Okinawa back to America...</td>
      <td>-1</td>
      <td>4.106157e-10</td>
    </tr>
    <tr>
      <th>70137</th>
      <td>Lulu Ladybug Rocker by Rockabye</td>
      <td>Here's my letter I sent to Rockabye:I purchase...</td>
      <td>1</td>
      <td>Heres my letter I sent to RockabyeI purchased ...</td>
      <td>-1</td>
      <td>4.367577e-10</td>
    </tr>
    <tr>
      <th>115108</th>
      <td>Blueberry Deluxe Diaper, Cow</td>
      <td>I purchased the Blueberry One Sized Bamboo Del...</td>
      <td>2</td>
      <td>I purchased the Blueberry One Sized Bamboo Del...</td>
      <td>-1</td>
      <td>9.164181e-10</td>
    </tr>
    <tr>
      <th>7075</th>
      <td>Peace of Mind Two 900 Mhz Baby Receivers, Monitor</td>
      <td>If we only knew when we registered how terribl...</td>
      <td>1</td>
      <td>If we only knew when we registered how terribl...</td>
      <td>-1</td>
      <td>1.019131e-09</td>
    </tr>
    <tr>
      <th>5033</th>
      <td>Playtex 3 Pack BPA Free VentAire Wide Bottles,...</td>
      <td>Initially, I thought these angled bottles make...</td>
      <td>1</td>
      <td>Initially I thought these angled bottles make ...</td>
      <td>-1</td>
      <td>1.154985e-09</td>
    </tr>
    <tr>
      <th>13712</th>
      <td>Badger Basket Elegance Round Baby Bassinet, Wh...</td>
      <td>I registered for this item &amp; it was ordered fo...</td>
      <td>1</td>
      <td>I registered for this item  it was ordered for...</td>
      <td>-1</td>
      <td>1.412743e-09</td>
    </tr>
    <tr>
      <th>3746</th>
      <td>Playtex Diaper Genie - First Refill Included</td>
      <td>Prior to parenthood, I had heard several paren...</td>
      <td>1</td>
      <td>Prior to parenthood I had heard several parent...</td>
      <td>-1</td>
      <td>1.505042e-09</td>
    </tr>
    <tr>
      <th>4835</th>
      <td>JJ Cole Premaxx Sling Carrier - New Edition Re...</td>
      <td>I bought this for my daughter while I was stil...</td>
      <td>1</td>
      <td>I bought this for my daughter while I was stil...</td>
      <td>-1</td>
      <td>1.595787e-09</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>74899</th>
      <td>Graco Blossom Highchair, Townsend</td>
      <td>We love this highchair.  We have a 4 year old ...</td>
      <td>5</td>
      <td>We love this highchair  We have a 4 year old a...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>162687</th>
      <td>Joovy Caboose Too Rear Seat, Greenie</td>
      <td>We are thrilled with this rear seat. This litt...</td>
      <td>5</td>
      <td>We are thrilled with this rear seat This littl...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>14008</th>
      <td>Stork Craft Beatrice Combo Tower Chest, White</td>
      <td>I bought the tower despite the bad reviews and...</td>
      <td>5</td>
      <td>I bought the tower despite the bad reviews and...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>135152</th>
      <td>Maxi-Cosi Pria 70 with Tiny Fit Convertible Ca...</td>
      <td>We've been using Britax for our boy (now 14 mo...</td>
      <td>5</td>
      <td>Weve been using Britax for our boy now 14 mont...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>108943</th>
      <td>Britax B-Ready Stroller, Black</td>
      <td>Some differences with Uppababy Vs. Britax B-Re...</td>
      <td>4</td>
      <td>Some differences with Uppababy Vs Britax BRead...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>108803</th>
      <td>Chicco Keyfit 22 Pound Infant Car Seat And Bas...</td>
      <td>I bought this right before the KeyFit 30 came ...</td>
      <td>5</td>
      <td>I bought this right before the KeyFit 30 came ...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>21557</th>
      <td>Joovy Caboose Stand On Tandem Stroller, Black</td>
      <td>Ok, I read all the reviews already posted here...</td>
      <td>5</td>
      <td>Ok I read all the reviews already posted here ...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>152013</th>
      <td>UPPAbaby Vista Stroller, Denny</td>
      <td>I researched strollers for months and months b...</td>
      <td>5</td>
      <td>I researched strollers for months and months b...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>111746</th>
      <td>Baby Jogger 2011 City Mini Double Stroller, Bl...</td>
      <td>Before purchasing this stroller, I read severa...</td>
      <td>5</td>
      <td>Before purchasing this stroller I read several...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>140780</th>
      <td>Diono RadianR100 Convertible Car Seat, Dune</td>
      <td>i bought this when the seat was owned by Sunsh...</td>
      <td>5</td>
      <td>i bought this when the seat was owned by Sunsh...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>59320</th>
      <td>Evenflo Aura Select Travel System - Caroline</td>
      <td>I'm about to be a first-time mom, so I spent w...</td>
      <td>5</td>
      <td>Im about to be a firsttime mom so I spent week...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>171041</th>
      <td>CTA Digital 2-in-1 iPotty with Activity Seat f...</td>
      <td>I'll just say in advance that the haters can j...</td>
      <td>5</td>
      <td>Ill just say in advance that the haters can ju...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>106455</th>
      <td>Quinny Senzz 2011 Fashion Stroller, Star</td>
      <td>I am very pleased overall with the Quinny Senz...</td>
      <td>4</td>
      <td>I am very pleased overall with the Quinny Senz...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>158209</th>
      <td>Ubbi Cloth Diaper Pail Liner</td>
      <td>(updated 3.22.13) After extensive research, tr...</td>
      <td>5</td>
      <td>updated 32213 After extensive research trial a...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>88659</th>
      <td>ERGObaby Original Baby Carrier, Galaxy Grey</td>
      <td>We purchased this carrier after a recommendati...</td>
      <td>5</td>
      <td>We purchased this carrier after a recommendati...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>88680</th>
      <td>ERGObaby Original Baby Carrier, Galaxy Grey</td>
      <td>After reading many online reviews, asking othe...</td>
      <td>5</td>
      <td>After reading many online reviews asking other...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>123632</th>
      <td>Zooper 2011 Waltz Standard Stroller, Flax Brown</td>
      <td>I did a TON of research before I purchased thi...</td>
      <td>5</td>
      <td>I did a TON of research before I purchased thi...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>93690</th>
      <td>The First Years Ignite Stroller</td>
      <td>The last thing we wanted was to purchase more ...</td>
      <td>5</td>
      <td>The last thing we wanted was to purchase more ...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>134265</th>
      <td>UPPAbaby Cruz Stroller, Denny</td>
      <td>We bought this stroller after selling our belo...</td>
      <td>5</td>
      <td>We bought this stroller after selling our belo...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>41763</th>
      <td>Kolcraft Contours Lite Stroller Plus with iPod...</td>
      <td>After considering several lightweight stroller...</td>
      <td>4</td>
      <td>After considering several lightweight stroller...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>147975</th>
      <td>Baby Jogger City Mini GT Single Stroller, Shad...</td>
      <td>Let me start by saying that I have gone throug...</td>
      <td>5</td>
      <td>Let me start by saying that I have gone throug...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>100166</th>
      <td>Infantino Wrap and Tie Baby Carrier, Black Blu...</td>
      <td>I bought this carrier when my daughter was abo...</td>
      <td>5</td>
      <td>I bought this carrier when my daughter was abo...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>167047</th>
      <td>Inglesina 2013 Trip Stroller, Lampone Purple</td>
      <td>I did many hours of research reading reviews a...</td>
      <td>5</td>
      <td>I did many hours of research reading reviews a...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>48158</th>
      <td>Phil &amp;amp; Ted's &amp;quot;2008 Version 2&amp;quot; Sp...</td>
      <td>We're keeping this stroller! After much resear...</td>
      <td>4</td>
      <td>Were keeping this stroller After much research...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>129212</th>
      <td>Baby Jogger 2011 City Select Stroller in Ameth...</td>
      <td>I have the Baby Jogger City Select with Second...</td>
      <td>5</td>
      <td>I have the Baby Jogger City Select with Second...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>166929</th>
      <td>Britax Pavilion 70-G3 Convertible Car Seat Sea...</td>
      <td>We LOVE this seat! As parents to 8 children ra...</td>
      <td>5</td>
      <td>We LOVE this seat As parents to 8 children ran...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>76549</th>
      <td>Britax Advocate 65 CS Click &amp;amp; Safe Convert...</td>
      <td>The Britax Advocate CS appears similar to the ...</td>
      <td>4</td>
      <td>The Britax Advocate CS appears similar to the ...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>50735</th>
      <td>Joovy Zoom 360 Swivel Wheel Jogging Stroller, ...</td>
      <td>The joovy zoom 360 was the perfect solution fo...</td>
      <td>5</td>
      <td>The joovy zoom 360 was the perfect solution fo...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>103186</th>
      <td>Thirsties Duo Wrap Snap, Ocean Blue, Size One ...</td>
      <td>I am reviewing these covers because reading re...</td>
      <td>5</td>
      <td>I am reviewing these covers because reading re...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>129722</th>
      <td>Bumbleride 2011 Flite Lightweight Compact Trav...</td>
      <td>This is a review of the 2012 Bumbleride Flite ...</td>
      <td>5</td>
      <td>This is a review of the 2012 Bumbleride Flite ...</td>
      <td>1</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
<p>50026 rows × 6 columns</p>
</div>



# Compute accuracy of the classifier


```python
test_matrix = vectorizer.transform(test_data['review_clean'].values.astype('U'))
scores_test = sentiment_model.predict(test_matrix)
print(scores_test)
```

    [ 1  1  1 ... -1  1  1]
    


```python
type(scores_test)
```




    numpy.ndarray




```python
test_data['sentiment']
```




    103733    1
    157975    1
    127693    1
    113718   -1
    20416     1
    62236    -1
    93741     1
    182819    1
    108942    1
    63171     1
    27972     1
    123316    1
    60874     1
    89381     1
    146248    1
    167897    1
    129689    1
    54150     1
    8352      1
    55700     1
    83479     1
    93618     1
    130762   -1
    73991     1
    174021    1
    17830     1
    62700     1
    33106     1
    156734   -1
    67422     1
             ..
    29152     1
    145868    1
    133924    1
    144259    1
    155251    1
    49516     1
    139738    1
    155993    1
    28007     1
    124413   -1
    114981    1
    115743   -1
    109348    1
    73490    -1
    10461     1
    120390   -1
    56823     1
    110189    1
    151692    1
    134663    1
    14881     1
    106787    1
    9918      1
    79920     1
    31318     1
    54722    -1
    106191    1
    79222    -1
    93173    -1
    86465     1
    Name: sentiment, Length: 50026, dtype: int64




```python
len(scores_test)
```




    50026




```python
len(test_data['sentiment'])
```




    50026




```python
type(test_data['sentiment'])
```




    pandas.core.series.Series




```python
test_data['sentiment'].index
```




    Int64Index([103733, 157975, 127693, 113718,  20416,  62236,  93741, 182819,
                108942,  63171,
                ...
                 14881, 106787,   9918,  79920,  31318,  54722, 106191,  79222,
                 93173,  86465],
               dtype='int64', length=50026)




```python
test_data['sentiment'].values
```




    array([ 1,  1,  1, ..., -1, -1,  1], dtype=int64)




```python
type(test_data['sentiment'].values)
```




    numpy.ndarray




```python
accuracy = (scores_test == test_data['sentiment'].values)
```


```python
accuracy
```




    array([ True,  True,  True, ...,  True, False,  True])




```python
len(accuracy)
```




    50026




```python
correct_counts = np.count_nonzero(accuracy == True)
```


```python
accuracy_math = correct_counts/len(accuracy)
```


```python
accuracy_math
```




    0.9300163914764322



# Learn another classifier with fewer words


```python
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
      'work', 'product', 'money', 'would', 'return']
```


```python
vectorizer_word_subset = CountVectorizer(vocabulary=significant_words) # limit to 20 words
train_matrix_word_subset = vectorizer_word_subset.fit_transform(train_data['review_clean'].values.astype('U'))
test_matrix_word_subset = vectorizer_word_subset.transform(test_data['review_clean'].values.astype('U'))
```

# Train a logistic regression model on a subset of data


```python
simple_model = linear_model.LogisticRegression()
```


```python
simple_model.fit(train_matrix_word_subset, train_data['sentiment'])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
simple_model.coef_
```




    array([[ 1.34240752,  0.93331879,  1.17057657,  0.09645538,  0.49546276,
             1.51007107,  1.70968884,  0.48996502,  0.18266183,  0.06213742,
            -1.63855688, -0.16152094, -0.52904125, -2.01345347, -2.36957333,
            -0.64400431, -0.32167306, -0.90190808, -0.33086295, -2.06984646]])




```python
simple_model.intercept_
```




    array([1.30862795])



## exclude the intercept (w0)

### ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
###      'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
###      'work', 'product', 'money', 'would', 'return']


```python
simple_model_noint = linear_model.LogisticRegression(fit_intercept=False)
```


```python
simple_model_noint.fit(train_matrix_word_subset, train_data['sentiment'])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=False,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
simple_model_noint.coef_
```




    array([[ 1.97893948,  1.5148022 ,  1.69499344,  0.49472526,  0.91975043,
             2.17222529,  2.41423013,  0.94998025,  0.4492522 ,  0.22932683,
            -1.41097026,  0.10142036, -0.27978831, -1.86577283, -2.13991375,
            -0.41593736,  0.01620127, -0.61563086, -0.06504618, -1.93940329]])




```python
simple_model_noint.intercept_
```




    0.0



# Comparing models

## First, compute the classification accuracy on the train_data


```python
def model_accuracy(model, input_matrix, target_parameter):
    recall = model.predict(input_matrix)
    TF_array = (recall == target_parameter)
    correct_counts = np.count_nonzero(TF_array == True)
    accu_math = correct_counts/len(TF_array)
    return accu_math
```


```python
model_accuracy(sentiment_model, train_matrix, train_data['sentiment'])
```




    0.9689529325086099




```python
model_accuracy(simple_model, train_matrix_word_subset, train_data['sentiment'])
```




    0.867373164504909




```python
model_accuracy(simple_model_noint, train_matrix_word_subset, train_data['sentiment'])
```




    0.7785411990473416



# ## First, compute the classification accuracy on the test_data


```python
model_accuracy(sentiment_model, test_matrix, test_data['sentiment'])
```




    0.9300163914764322




```python
model_accuracy(simple_model, test_matrix_word_subset, test_data['sentiment'])
```




    0.867269020109543




```python
model_accuracy(simple_model_noint, test_matrix_word_subset, test_data['sentiment'])
```




    0.7796545796185983



## test numpy sort function here


```python
list1 = [[4,3,2],[2,1,4],[1,1,2],[4,1,3]]
array=np.array(list1)
```


```python
array
```




    array([[4, 3, 2],
           [2, 1, 4],
           [1, 1, 2],
           [4, 1, 3]])




```python
array.sort(axis=1)
```


```python
array
```




    array([[2, 3, 4],
           [1, 2, 4],
           [1, 1, 2],
           [1, 3, 4]])




```python
array.sort(axis=0)
```


```python
array
```




    array([[1, 1, 2],
           [1, 2, 4],
           [1, 3, 4],
           [2, 3, 4]])




```python
len(array)
```




    4




```python
array[-3:]
```




    array([[1, 2, 4],
           [1, 3, 4],
           [2, 3, 4]])



# Test ndarrary to pandas series


```python
test_data = proba
```


```python
test_data
```




    array([[3.42077940e-03, 9.96579221e-01],
           [1.22103669e-07, 9.99999878e-01],
           [2.21610865e-02, 9.77838914e-01]])




```python
type(test_data)
```




    numpy.ndarray



2nd ([0] -> [1]) element of all the rows


```python
test_positive_possibility = test_data[:,1]
```


```python
test_positive_possibility
```




    array([0.99657922, 0.99999988, 0.97783891])




```python
type(sample_test_data)
```




    pandas.core.frame.DataFrame




```python
sample_test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>review_clean</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27972</th>
      <td>Bumkins Waterproof Superbib, Blue Fizz</td>
      <td>I love these bibs!  We have about 8 of them in...</td>
      <td>5</td>
      <td>I love these bibs  We have about 8 of them in ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>123316</th>
      <td>Safety 1st High Door Lock</td>
      <td>My 4 year old gets up earlier than me, this me...</td>
      <td>5</td>
      <td>My 4 year old gets up earlier than me this mea...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>60874</th>
      <td>DadGear Courier Diaper Bag - Orange Retro Stripe</td>
      <td>love the bag, especially since it's over the s...</td>
      <td>4</td>
      <td>love the bag especially since its over the sho...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample_test_data.loc[:,'probability'] = test_positive_possibility
```

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\indexing.py:357: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[key] = _infer_fill_value(value)
    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\indexing.py:537: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self.obj[item] = s
    


```python
# sample_test_data['probability'] = test_positive_possibility
```


```python
sample_test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>review_clean</th>
      <th>sentiment</th>
      <th>probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27972</th>
      <td>Bumkins Waterproof Superbib, Blue Fizz</td>
      <td>I love these bibs!  We have about 8 of them in...</td>
      <td>5</td>
      <td>I love these bibs  We have about 8 of them in ...</td>
      <td>1</td>
      <td>0.996579</td>
    </tr>
    <tr>
      <th>123316</th>
      <td>Safety 1st High Door Lock</td>
      <td>My 4 year old gets up earlier than me, this me...</td>
      <td>5</td>
      <td>My 4 year old gets up earlier than me this mea...</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>60874</th>
      <td>DadGear Courier Diaper Bag - Orange Retro Stripe</td>
      <td>love the bag, especially since it's over the s...</td>
      <td>4</td>
      <td>love the bag especially since its over the sho...</td>
      <td>1</td>
      <td>0.977839</td>
    </tr>
  </tbody>
</table>
</div>



## define series index and columns


```python
df = DataFrame(test_data,
              index = ['one', 'two', 'three'],
              columns = ['negative', 'positive'])
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>negative</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>3.420779e-03</td>
      <td>0.996579</td>
    </tr>
    <tr>
      <th>two</th>
      <td>1.221037e-07</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>three</th>
      <td>2.216109e-02</td>
      <td>0.977839</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample_test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>review</th>
      <th>rating</th>
      <th>review_clean</th>
      <th>sentiment</th>
      <th>probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27972</th>
      <td>Bumkins Waterproof Superbib, Blue Fizz</td>
      <td>I love these bibs!  We have about 8 of them in...</td>
      <td>5</td>
      <td>I love these bibs  We have about 8 of them in ...</td>
      <td>1</td>
      <td>0.996579</td>
    </tr>
    <tr>
      <th>123316</th>
      <td>Safety 1st High Door Lock</td>
      <td>My 4 year old gets up earlier than me, this me...</td>
      <td>5</td>
      <td>My 4 year old gets up earlier than me this mea...</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>60874</th>
      <td>DadGear Courier Diaper Bag - Orange Retro Stripe</td>
      <td>love the bag, especially since it's over the s...</td>
      <td>4</td>
      <td>love the bag especially since its over the sho...</td>
      <td>1</td>
      <td>0.977839</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_1 = DataFrame(test_data,
              columns = ['negative', 'positive'])
```


```python
df_1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>negative</th>
      <th>positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.420779e-03</td>
      <td>0.996579</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.221037e-07</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.216109e-02</td>
      <td>0.977839</td>
    </tr>
  </tbody>
</table>
</div>


