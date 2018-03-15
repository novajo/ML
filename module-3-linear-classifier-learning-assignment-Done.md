
# Implementing logistic regression from scratch

The goal of this notebook is to implement your own logistic regression classifier. You will:

 * Extract features from Amazon product reviews.
 * Convert an SFrame into a NumPy array.
 * Implement the link function for logistic regression.
 * Write a function to compute the derivative of the log likelihood function with respect to a single coefficient.
 * Implement gradient ascent.
 * Given a set of coefficients, predict sentiments.
 * Compute classification accuracy for the logistic regression model.
 
Let's get started!
    
## Fire up numpy, pandas and sklearn


```python
# import graphlab
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import string
```

## Load review dataset

For this assignment, we will use a subset of the Amazon product review dataset. The subset was chosen to contain similar numbers of positive and negative reviews, as the original dataset consisted primarily of positive reviews.


```python
products = pd.read_csv("D:/ml_data/amazon_baby_subset.csv")
```


```python
len(products)
```




    53072




```python
# products = products[26000:27000]
```


```python
products = products.fillna({'review': ''})
```

One column of this dataset is 'sentiment', corresponding to the class label with +1 indicating a review with positive sentiment and -1 indicating one with negative sentiment.


```python
products['sentiment']
```




    0        1
    1        1
    2        1
    3        1
    4        1
    5        1
    6        1
    7        1
    8        1
    9        1
    10       1
    11       1
    12       1
    13       1
    14       1
    15       1
    16       1
    17       1
    18       1
    19       1
    20       1
    21       1
    22       1
    23       1
    24       1
    25       1
    26       1
    27       1
    28       1
    29       1
            ..
    53042   -1
    53043   -1
    53044   -1
    53045   -1
    53046   -1
    53047   -1
    53048   -1
    53049   -1
    53050   -1
    53051   -1
    53052   -1
    53053   -1
    53054   -1
    53055   -1
    53056   -1
    53057   -1
    53058   -1
    53059   -1
    53060   -1
    53061   -1
    53062   -1
    53063   -1
    53064   -1
    53065   -1
    53066   -1
    53067   -1
    53068   -1
    53069   -1
    53070   -1
    53071   -1
    Name: sentiment, Length: 53072, dtype: int64



Let us quickly explore more of this dataset.  The 'name' column indicates the name of the product.  Here we list the first 10 products in the dataset.  We then count the number of positive and negative reviews.


```python
products.head(10)['name']
```




    0    Stop Pacifier Sucking without tears with Thumb...
    1      Nature's Lullabies Second Year Sticker Calendar
    2      Nature's Lullabies Second Year Sticker Calendar
    3                          Lamaze Peekaboo, I Love You
    4    SoftPlay Peek-A-Boo Where's Elmo A Children's ...
    5                            Our Baby Girl Memory Book
    6    Hunnt&reg; Falling Flowers and Birds Kids Nurs...
    7    Blessed By Pope Benedict XVI Divine Mercy Full...
    8    Cloth Diaper Pins Stainless Steel Traditional ...
    9    Cloth Diaper Pins Stainless Steel Traditional ...
    Name: name, dtype: object




```python
print('# of positive reviews ={}'.format(str(len(products[products['sentiment']==1]))))
print('# of negative reviews ={}'.format(str(len(products[products['sentiment']==-1]))))
```

    # of positive reviews =26579
    # of negative reviews =26493
    

**Note:** For this assignment, we eliminated class imbalance by choosing 
a subset of the data with a similar number of positive and negative reviews. 

## Apply text cleaning on the review data

In this section, we will perform some simple feature cleaning using **SFrames**. The last assignment used all words in building bag-of-words features, but here we limit ourselves to 193 words (for simplicity). We compiled a list of 193 most frequent words into a JSON file. 

Now, we will load these words from this JSON file:


```python
import json
with open('important_words.json', 'r') as f: # Reads the list of most frequent words
    important_words = json.load(f)
important_words = [str(s) for s in important_words]
```


```python
print(important_words)
```

    ['baby', 'one', 'great', 'love', 'use', 'would', 'like', 'easy', 'little', 'seat', 'old', 'well', 'get', 'also', 'really', 'son', 'time', 'bought', 'product', 'good', 'daughter', 'much', 'loves', 'stroller', 'put', 'months', 'car', 'still', 'back', 'used', 'recommend', 'first', 'even', 'perfect', 'nice', 'bag', 'two', 'using', 'got', 'fit', 'around', 'diaper', 'enough', 'month', 'price', 'go', 'could', 'soft', 'since', 'buy', 'room', 'works', 'made', 'child', 'keep', 'size', 'small', 'need', 'year', 'big', 'make', 'take', 'easily', 'think', 'crib', 'clean', 'way', 'quality', 'thing', 'better', 'without', 'set', 'new', 'every', 'cute', 'best', 'bottles', 'work', 'purchased', 'right', 'lot', 'side', 'happy', 'comfortable', 'toy', 'able', 'kids', 'bit', 'night', 'long', 'fits', 'see', 'us', 'another', 'play', 'day', 'money', 'monitor', 'tried', 'thought', 'never', 'item', 'hard', 'plastic', 'however', 'disappointed', 'reviews', 'something', 'going', 'pump', 'bottle', 'cup', 'waste', 'return', 'amazon', 'different', 'top', 'want', 'problem', 'know', 'water', 'try', 'received', 'sure', 'times', 'chair', 'find', 'hold', 'gate', 'open', 'bottom', 'away', 'actually', 'cheap', 'worked', 'getting', 'ordered', 'came', 'milk', 'bad', 'part', 'worth', 'found', 'cover', 'many', 'design', 'looking', 'weeks', 'say', 'wanted', 'look', 'place', 'purchase', 'looks', 'second', 'piece', 'box', 'pretty', 'trying', 'difficult', 'together', 'though', 'give', 'started', 'anything', 'last', 'company', 'come', 'returned', 'maybe', 'took', 'broke', 'makes', 'stay', 'instead', 'idea', 'head', 'said', 'less', 'went', 'working', 'high', 'unit', 'seems', 'picture', 'completely', 'wish', 'buying', 'babies', 'won', 'tub', 'almost', 'either']
    


```python
len(important_words)
```




    193



Now, we will perform 2 simple data transformations:

1. Remove punctuation using [Python's built-in](https://docs.python.org/2/library/string.html) string functionality.
2. Compute word counts (only for **important_words**)

We start with *Step 1* which can be done as follows:


```python
def remove_punctuation(text):
    map = str.maketrans('', '', string.punctuation)
    return text.translate(map)

def remove_punctuations(s):
    print(type(s))
    table = str.maketrans({key: None for key in string.punctuation})
    new_s = s.translate(table)
    return new_s
```


```python
# products['review_clean'] = products['review'].str.replace('[^\w\s]', '')
```


```python
# products['review'][450:460]
```


```python
products['review_clean'] = products['review'].apply(remove_punctuation)
```


```python
products['review_clean'].head()
```




    0    All of my kids have cried nonstop when I tried...
    1    We wanted to get something to keep track of ou...
    2    My daughter had her 1st baby over a year ago S...
    3    One of babys first and favorite books and it i...
    4    Very cute interactive book My son loves this b...
    Name: review_clean, dtype: object




```python
products.head()
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
      <th>sentiment</th>
      <th>review_clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>All of my kids have cried non-stop when I trie...</td>
      <td>5</td>
      <td>1</td>
      <td>All of my kids have cried nonstop when I tried...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>We wanted to get something to keep track of ou...</td>
      <td>5</td>
      <td>1</td>
      <td>We wanted to get something to keep track of ou...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>My daughter had her 1st baby over a year ago. ...</td>
      <td>5</td>
      <td>1</td>
      <td>My daughter had her 1st baby over a year ago S...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lamaze Peekaboo, I Love You</td>
      <td>One of baby's first and favorite books, and it...</td>
      <td>4</td>
      <td>1</td>
      <td>One of babys first and favorite books and it i...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SoftPlay Peek-A-Boo Where's Elmo A Children's ...</td>
      <td>Very cute interactive book! My son loves this ...</td>
      <td>5</td>
      <td>1</td>
      <td>Very cute interactive book My son loves this b...</td>
    </tr>
  </tbody>
</table>
</div>



Now we proceed with *Step 2*. For each word in **important_words**, we compute a count for the number of times the word occurs in the review. We will store this count in a separate column (one for each word). The result of this feature processing is a single column for each word in **important_words** which keeps a count of the number of times the respective word occurs in the review text.


**Note:** There are several ways of doing this. In this assignment, we use the built-in *count* function for Python lists. Each review string is first split into individual words and the number of occurances of a given word is counted.


```python
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : s.split().count(word))
```

The SFrame **products** now contains one column for each of the 193 **important_words**. As an example, the column **perfect** contains a count of the number of times the word **perfect** occurs in each of the reviews.


```python
products['perfect']
```




    0        0
    1        0
    2        0
    3        1
    4        0
    5        0
    6        0
    7        0
    8        0
    9        0
    10       0
    11       1
    12       0
    13       0
    14       0
    15       0
    16       0
    17       0
    18       0
    19       0
    20       0
    21       0
    22       1
    23       0
    24       1
    25       0
    26       0
    27       1
    28       0
    29       0
            ..
    53042    0
    53043    0
    53044    0
    53045    0
    53046    0
    53047    0
    53048    0
    53049    0
    53050    0
    53051    1
    53052    0
    53053    0
    53054    1
    53055    0
    53056    0
    53057    0
    53058    0
    53059    0
    53060    0
    53061    0
    53062    0
    53063    0
    53064    0
    53065    0
    53066    0
    53067    0
    53068    0
    53069    0
    53070    0
    53071    0
    Name: perfect, Length: 53072, dtype: int64



Now, write some code to compute the number of product reviews that contain the word **perfect**.

**Hint**: 
* First create a column called `contains_perfect` which is set to 1 if the count of the word **perfect** (stored in column **perfect**) is >= 1.
* Sum the number of 1s in the column `contains_perfect`.


```python
products['contains_perfect'] = products['perfect'].apply(lambda x: 1 if x >= 1 else 0)
```


```python
products
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
      <th>sentiment</th>
      <th>review_clean</th>
      <th>baby</th>
      <th>one</th>
      <th>great</th>
      <th>love</th>
      <th>use</th>
      <th>...</th>
      <th>picture</th>
      <th>completely</th>
      <th>wish</th>
      <th>buying</th>
      <th>babies</th>
      <th>won</th>
      <th>tub</th>
      <th>almost</th>
      <th>either</th>
      <th>contains_perfect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>All of my kids have cried non-stop when I trie...</td>
      <td>5</td>
      <td>1</td>
      <td>All of my kids have cried nonstop when I tried...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>We wanted to get something to keep track of ou...</td>
      <td>5</td>
      <td>1</td>
      <td>We wanted to get something to keep track of ou...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>My daughter had her 1st baby over a year ago. ...</td>
      <td>5</td>
      <td>1</td>
      <td>My daughter had her 1st baby over a year ago S...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lamaze Peekaboo, I Love You</td>
      <td>One of baby's first and favorite books, and it...</td>
      <td>4</td>
      <td>1</td>
      <td>One of babys first and favorite books and it i...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SoftPlay Peek-A-Boo Where's Elmo A Children's ...</td>
      <td>Very cute interactive book! My son loves this ...</td>
      <td>5</td>
      <td>1</td>
      <td>Very cute interactive book My son loves this b...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Our Baby Girl Memory Book</td>
      <td>Beautiful book, I love it to record cherished ...</td>
      <td>5</td>
      <td>1</td>
      <td>Beautiful book I love it to record cherished t...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hunnt&amp;reg; Falling Flowers and Birds Kids Nurs...</td>
      <td>Try this out for a spring project !Easy ,fun a...</td>
      <td>5</td>
      <td>1</td>
      <td>Try this out for a spring project Easy fun and...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Blessed By Pope Benedict XVI Divine Mercy Full...</td>
      <td>very nice Divine Mercy Pendant of Jesus now on...</td>
      <td>5</td>
      <td>1</td>
      <td>very nice Divine Mercy Pendant of Jesus now on...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cloth Diaper Pins Stainless Steel Traditional ...</td>
      <td>We bought the pins as my 6 year old Autistic s...</td>
      <td>4</td>
      <td>1</td>
      <td>We bought the pins as my 6 year old Autistic s...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cloth Diaper Pins Stainless Steel Traditional ...</td>
      <td>It has been many years since we needed diaper ...</td>
      <td>5</td>
      <td>1</td>
      <td>It has been many years since we needed diaper ...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Newborn Baby Tracker&amp;reg; - Round the Clock Ch...</td>
      <td>We found this book at a rummage sale and found...</td>
      <td>5</td>
      <td>1</td>
      <td>We found this book at a rummage sale and found...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Newborn Baby Tracker&amp;reg; - Round the Clock Ch...</td>
      <td>I'm a new mom and I was looking for something ...</td>
      <td>5</td>
      <td>1</td>
      <td>Im a new mom and I was looking for something t...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Newborn Baby Tracker&amp;reg; - Round the Clock Ch...</td>
      <td>I loved how this book was set up to keep track...</td>
      <td>5</td>
      <td>1</td>
      <td>I loved how this book was set up to keep track...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Newborn Baby Tracker&amp;reg; - Round the Clock Ch...</td>
      <td>I received this at my baby shower and it has b...</td>
      <td>5</td>
      <td>1</td>
      <td>I received this at my baby shower and it has b...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Neurosmith - Music Blocks with Mozart Music Ca...</td>
      <td>My daughter started playing with her Music Blo...</td>
      <td>5</td>
      <td>1</td>
      <td>My daughter started playing with her Music Blo...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Neurosmith - Music Blocks with Mozart Music Ca...</td>
      <td>It takes a youthful spirit of inquiry and fasc...</td>
      <td>5</td>
      <td>1</td>
      <td>It takes a youthful spirit of inquiry and fasc...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Neurosmith - Music Blocks with Mozart Music Ca...</td>
      <td>This is an interesting and educational toy.  I...</td>
      <td>4</td>
      <td>1</td>
      <td>This is an interesting and educational toy  I ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Neurosmith - Music Blocks with Mozart Music Ca...</td>
      <td>Recently I have purchased the musical mozart b...</td>
      <td>5</td>
      <td>1</td>
      <td>Recently I have purchased the musical mozart b...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>my first fish bowl by lamaze / learning curve</td>
      <td>We first bought this toy for our oldest child ...</td>
      <td>5</td>
      <td>1</td>
      <td>We first bought this toy for our oldest child ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Pedal Farm Tractor</td>
      <td>We bought this tractor for our 2 and a half-ye...</td>
      <td>5</td>
      <td>1</td>
      <td>We bought this tractor for our 2 and a halfyea...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>This is a great toy.  The wheels really work a...</td>
      <td>5</td>
      <td>1</td>
      <td>This is a great toy  The wheels really work an...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>After being inundated with toys that require b...</td>
      <td>5</td>
      <td>1</td>
      <td>After being inundated with toys that require b...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>We bought these trucks for our 15 mo. old daug...</td>
      <td>5</td>
      <td>1</td>
      <td>We bought these trucks for our 15 mo old daugh...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>For well over a year my son has enjoyed stacki...</td>
      <td>5</td>
      <td>1</td>
      <td>For well over a year my son has enjoyed stacki...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>We just bought this for my 9 month old daughte...</td>
      <td>5</td>
      <td>1</td>
      <td>We just bought this for my 9 month old daughte...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>This is a wonderful toy that is fun, education...</td>
      <td>5</td>
      <td>1</td>
      <td>This is a wonderful toy that is fun educationa...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Sassy Who Loves Baby? Photo Album Book with te...</td>
      <td>I bought this for a new granddaughter.  I will...</td>
      <td>5</td>
      <td>1</td>
      <td>I bought this for a new granddaughter  I will ...</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Earlyears: Earl E. Bird with Teething Rings</td>
      <td>We received an Earl E. Bird as a gift when we ...</td>
      <td>5</td>
      <td>1</td>
      <td>We received an Earl E Bird as a gift when we h...</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Earlyears: Earl E. Bird with Teething Rings</td>
      <td>This little toy is safe for infants, and offer...</td>
      <td>5</td>
      <td>1</td>
      <td>This little toy is safe for infants and offers...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>My Quiet Book, Fabric Activity Book for Children</td>
      <td>This is exactly like the one I had when I was ...</td>
      <td>5</td>
      <td>1</td>
      <td>This is exactly like the one I had when I was ...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>53042</th>
      <td>Beautiful Luxury Nightcap Baby Suspender Baby ...</td>
      <td>I only put 1 star because I had to. The photos...</td>
      <td>1</td>
      <td>-1</td>
      <td>I only put 1 star because I had to The photos ...</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53043</th>
      <td>Beautiful Luxury Nightcap Baby Suspender Baby ...</td>
      <td>Amazon, please take this down. The pictures sh...</td>
      <td>1</td>
      <td>-1</td>
      <td>Amazon please take this down The pictures show...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53044</th>
      <td>Blackcell 2pairs Infant Toddler Baby Knee Pad ...</td>
      <td>Not very impressed. They are cute, however the...</td>
      <td>1</td>
      <td>-1</td>
      <td>Not very impressed They are cute however they ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53045</th>
      <td>Nuby Soft Spout Easy Grip Sippy Cup - 4 pk.</td>
      <td>I agree with the other 2 posters. These sippy ...</td>
      <td>1</td>
      <td>-1</td>
      <td>I agree with the other 2 posters These sippy c...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53046</th>
      <td>VTech Communications Safe and Sound Digital Au...</td>
      <td>These monitors do not work at all, I even atte...</td>
      <td>1</td>
      <td>-1</td>
      <td>These monitors do not work at all I even attem...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53047</th>
      <td>Giftoyou(TM) High Quality Lightweight Retracta...</td>
      <td>Our family purchased numerous of these because...</td>
      <td>1</td>
      <td>-1</td>
      <td>Our family purchased numerous of these because...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53048</th>
      <td>Giftoyou(TM) High Quality Lightweight Retracta...</td>
      <td>I was so happy when i bought this charger for ...</td>
      <td>1</td>
      <td>-1</td>
      <td>I was so happy when i bought this charger for ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53049</th>
      <td>zTcase&amp;trade; Bluetooth Wireless Keyboard Case...</td>
      <td>I was very excited to receive this keyboard ca...</td>
      <td>1</td>
      <td>-1</td>
      <td>I was very excited to receive this keyboard ca...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53050</th>
      <td>Geleeo Universal Stroller Cooling Pad - Pink</td>
      <td>I agree with the reviewer below, this pad you ...</td>
      <td>2</td>
      <td>-1</td>
      <td>I agree with the reviewer below this pad you w...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53051</th>
      <td>4Moms mamaRoo Plush Infant Seat - Silver</td>
      <td>As a designer, I loved the look of the Mommaro...</td>
      <td>1</td>
      <td>-1</td>
      <td>As a designer I loved the look of the Mommaroo...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53052</th>
      <td>4Moms mamaRoo Plush Infant Seat - Silver</td>
      <td>A friend told us to get this, said it was a &amp;#...</td>
      <td>2</td>
      <td>-1</td>
      <td>A friend told us to get this said it was a 34m...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53053</th>
      <td>Merry Muscles Ergonomic Jumper Exerciser Baby ...</td>
      <td>once in this thing, my 2mo. son loves this... ...</td>
      <td>2</td>
      <td>-1</td>
      <td>once in this thing my 2mo son loves this howev...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53054</th>
      <td>[Different Colors Available] Newisland&amp;reg; 79...</td>
      <td>I received this product free for review but de...</td>
      <td>2</td>
      <td>-1</td>
      <td>I received this product free for review but de...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53055</th>
      <td>Vandot for Samsung Galaxy S4 / I9500 ULTRA SLI...</td>
      <td>Omg this case was so ugly and so hard that I t...</td>
      <td>1</td>
      <td>-1</td>
      <td>Omg this case was so ugly and so hard that I t...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53056</th>
      <td>Nicerocker big Cute Sweet Kids Baby Girls Love...</td>
      <td>I only ordered one of this item and received 2...</td>
      <td>1</td>
      <td>-1</td>
      <td>I only ordered one of this item and received 2...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53057</th>
      <td>Clevamama 3-in-1 Sleep, Sit and Play Travel Ma...</td>
      <td>The mattress is supposedly hypoallergenic clev...</td>
      <td>1</td>
      <td>-1</td>
      <td>The mattress is supposedly hypoallergenic clev...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53058</th>
      <td>Graco Argos 65 3-in-1 Harness Booster, Link</td>
      <td>Short story, I was very disappointed with the ...</td>
      <td>1</td>
      <td>-1</td>
      <td>Short story I was very disappointed with the q...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53059</th>
      <td>K&amp;amp;C Baby Bath Seat Support Sling Shower Me...</td>
      <td>Absolute rip off!!! Not impressed at all this ...</td>
      <td>1</td>
      <td>-1</td>
      <td>Absolute rip off Not impressed at all this was...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53060</th>
      <td>Aqueduck Faucet Extender &amp;amp; Handle Extended...</td>
      <td>I wish I had bought the faucet extender and ha...</td>
      <td>2</td>
      <td>-1</td>
      <td>I wish I had bought the faucet extender and ha...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53061</th>
      <td>ANDI ROSE Baby Toddlers Floral Printed Ruffle ...</td>
      <td>Definitely made extremely poorly in china. Is ...</td>
      <td>1</td>
      <td>-1</td>
      <td>Definitely made extremely poorly in china Is n...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53062</th>
      <td>Vandot 2 in1 Accessory Set 3D Leather Case Lit...</td>
      <td>Cute but cheaply made.. The part where you put...</td>
      <td>2</td>
      <td>-1</td>
      <td>Cute but cheaply made The part where you put y...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53063</th>
      <td>Umai Authentic Hazelwood and CHERRY RAW (Unpol...</td>
      <td>Made no difference :/</td>
      <td>1</td>
      <td>-1</td>
      <td>Made no difference</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53064</th>
      <td>360 Degree Rotating Cover Case for Samsung Gal...</td>
      <td>Be careful it stains your screen protector.  B...</td>
      <td>1</td>
      <td>-1</td>
      <td>Be careful it stains your screen protector  Bo...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53065</th>
      <td>Summer Infant Pop 'n Play Portable Playard</td>
      <td>Good idea but too dangerous. I really wanted t...</td>
      <td>2</td>
      <td>-1</td>
      <td>Good idea but too dangerous I really wanted to...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53066</th>
      <td>Freeens Cool Seat Liner Breathing with 3d Mesh...</td>
      <td>It doesn't stay input. My daughter was sliding...</td>
      <td>1</td>
      <td>-1</td>
      <td>It doesnt stay input My daughter was sliding o...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53067</th>
      <td>Samsung Baby Care Washer, Stainless Platinum, ...</td>
      <td>My infant goes to a really crappy daycare, and...</td>
      <td>1</td>
      <td>-1</td>
      <td>My infant goes to a really crappy daycare and ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53068</th>
      <td>Mud Pie Milestone Stickers, Boy</td>
      <td>Pretty please open and inspect these stickers ...</td>
      <td>1</td>
      <td>-1</td>
      <td>Pretty please open and inspect these stickers ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53069</th>
      <td>Best BIB for Baby - Soft Bib (Pink-Elephant)</td>
      <td>Great 5-Star Product but An Obvious knock-off ...</td>
      <td>1</td>
      <td>-1</td>
      <td>Great 5Star Product but An Obvious knockoff of...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53070</th>
      <td>Bouncy&amp;reg; Inflatable Real Feel Hopping Cow</td>
      <td>When I received the item my initial thought wa...</td>
      <td>2</td>
      <td>-1</td>
      <td>When I received the item my initial thought wa...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53071</th>
      <td>Maxboost iPhone 5S/5 Case - Protective Snap-on...</td>
      <td>I got this case in the mail today, it came on ...</td>
      <td>2</td>
      <td>-1</td>
      <td>I got this case in the mail today it came on t...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>53072 rows × 199 columns</p>
</div>




```python
products['contains_perfect'].value_counts()
```




    0    50117
    1     2955
    Name: contains_perfect, dtype: int64



**Quiz Question**. How many reviews contain the word **perfect**?

## Convert SFrame to NumPy array

As you have seen previously, NumPy is a powerful library for doing matrix manipulation. Let us convert our data to matrices and then implement our algorithms with matrices.

First, make sure you can perform the following import.


```python
# import numpy as np
```

We now provide you with a function that extracts columns from an SFrame and converts them into a NumPy array. Two arrays are returned: one representing features and another representing class labels. Note that the feature matrix includes an additional column 'intercept' to take account of the intercept term.


```python
# def get_numpy_data(data_sframe, features, label):
#     data_sframe['intercept'] = 1
#     features = ['intercept'] + features
#     features_sframe = data_sframe[features]
#     feature_matrix = features_sframe.to_numpy()
#     label_sarray = data_sframe[label]
#     label_array = label_sarray.to_numpy()
#     return(feature_matrix, label_array)
```

Let us convert the data into NumPy arrays.


```python
# Warning: This may take a few minutes...
# feature_matrix, sentiment = get_numpy_data(products, important_words, 'sentiment') 
```

**Are you running this notebook on an Amazon EC2 t2.micro instance?** (If you are using your own machine, please skip this section)

It has been reported that t2.micro instances do not provide sufficient power to complete the conversion in acceptable amount of time. For interest of time, please refrain from running `get_numpy_data` function. Instead, download the [binary file](https://s3.amazonaws.com/static.dato.com/files/coursera/course-3/numpy-arrays/module-3-assignment-numpy-arrays.npz) containing the four NumPy arrays you'll need for the assignment. To load the arrays, run the following commands:
```
arrays = np.load('module-3-assignment-numpy-arrays.npz')
feature_matrix, sentiment = arrays['feature_matrix'], arrays['sentiment']
```


```python
sentiment = products['sentiment']
```


```python
feature_names = np.array(products.columns.tolist())
```


```python
unfeatured_columns = ['name', 'review', 'rating', 'sentiment', 'review_clean', 'contains_perfect']
```


```python
feature_matrix = products.drop(unfeatured_columns, axis=1, inplace=False)
```


```python
products.head()
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
      <th>sentiment</th>
      <th>review_clean</th>
      <th>baby</th>
      <th>one</th>
      <th>great</th>
      <th>love</th>
      <th>use</th>
      <th>...</th>
      <th>picture</th>
      <th>completely</th>
      <th>wish</th>
      <th>buying</th>
      <th>babies</th>
      <th>won</th>
      <th>tub</th>
      <th>almost</th>
      <th>either</th>
      <th>contains_perfect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>All of my kids have cried non-stop when I trie...</td>
      <td>5</td>
      <td>1</td>
      <td>All of my kids have cried nonstop when I tried...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>We wanted to get something to keep track of ou...</td>
      <td>5</td>
      <td>1</td>
      <td>We wanted to get something to keep track of ou...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>My daughter had her 1st baby over a year ago. ...</td>
      <td>5</td>
      <td>1</td>
      <td>My daughter had her 1st baby over a year ago S...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lamaze Peekaboo, I Love You</td>
      <td>One of baby's first and favorite books, and it...</td>
      <td>4</td>
      <td>1</td>
      <td>One of babys first and favorite books and it i...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SoftPlay Peek-A-Boo Where's Elmo A Children's ...</td>
      <td>Very cute interactive book! My son loves this ...</td>
      <td>5</td>
      <td>1</td>
      <td>Very cute interactive book My son loves this b...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 199 columns</p>
</div>




```python
idx = 0
new_col = None# can be a list, a Series, an array or a scalar   
feature_matrix.insert(loc=idx, column='intercept', value=new_col)
```


```python
feature_matrix['intercept'] = 1
```


```python
type(feature_matrix)
```




    pandas.core.frame.DataFrame




```python
feature_matrix
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
      <th>intercept</th>
      <th>baby</th>
      <th>one</th>
      <th>great</th>
      <th>love</th>
      <th>use</th>
      <th>would</th>
      <th>like</th>
      <th>easy</th>
      <th>little</th>
      <th>...</th>
      <th>seems</th>
      <th>picture</th>
      <th>completely</th>
      <th>wish</th>
      <th>buying</th>
      <th>babies</th>
      <th>won</th>
      <th>tub</th>
      <th>almost</th>
      <th>either</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>53042</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53043</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53044</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53045</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53046</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53047</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53048</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53049</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53050</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53051</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53052</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53053</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53054</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53055</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53056</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53057</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53058</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53059</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53060</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53061</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53062</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53063</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53064</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53065</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53066</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53067</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53068</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53069</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53070</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>53071</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>53072 rows × 194 columns</p>
</div>



** Quiz Question:** How many features are there in the **feature_matrix**?

** Quiz Question:** Assuming that the intercept is present, how does the number of features in **feature_matrix** relate to the number of features in the logistic regression model?

Now, let us see what the **sentiment** column looks like:


```python
sentiment
```




    0        1
    1        1
    2        1
    3        1
    4        1
    5        1
    6        1
    7        1
    8        1
    9        1
    10       1
    11       1
    12       1
    13       1
    14       1
    15       1
    16       1
    17       1
    18       1
    19       1
    20       1
    21       1
    22       1
    23       1
    24       1
    25       1
    26       1
    27       1
    28       1
    29       1
            ..
    53042   -1
    53043   -1
    53044   -1
    53045   -1
    53046   -1
    53047   -1
    53048   -1
    53049   -1
    53050   -1
    53051   -1
    53052   -1
    53053   -1
    53054   -1
    53055   -1
    53056   -1
    53057   -1
    53058   -1
    53059   -1
    53060   -1
    53061   -1
    53062   -1
    53063   -1
    53064   -1
    53065   -1
    53066   -1
    53067   -1
    53068   -1
    53069   -1
    53070   -1
    53071   -1
    Name: sentiment, Length: 53072, dtype: int64




```python
sentiment == +1
```




    0         True
    1         True
    2         True
    3         True
    4         True
    5         True
    6         True
    7         True
    8         True
    9         True
    10        True
    11        True
    12        True
    13        True
    14        True
    15        True
    16        True
    17        True
    18        True
    19        True
    20        True
    21        True
    22        True
    23        True
    24        True
    25        True
    26        True
    27        True
    28        True
    29        True
             ...  
    53042    False
    53043    False
    53044    False
    53045    False
    53046    False
    53047    False
    53048    False
    53049    False
    53050    False
    53051    False
    53052    False
    53053    False
    53054    False
    53055    False
    53056    False
    53057    False
    53058    False
    53059    False
    53060    False
    53061    False
    53062    False
    53063    False
    53064    False
    53065    False
    53066    False
    53067    False
    53068    False
    53069    False
    53070    False
    53071    False
    Name: sentiment, Length: 53072, dtype: bool



## Estimating conditional probability with link function

Recall from lecture that the link function is given by:
$$
P(y_i = +1 | \mathbf{x}_i,\mathbf{w}) = \frac{1}{1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))},
$$

where the feature vector $h(\mathbf{x}_i)$ represents the word counts of **important_words** in the review  $\mathbf{x}_i$. Complete the following function that implements the link function:


```python
'''
produces probablistic estimate for P(y_i = +1 | x_i, w).
estimate ranges between 0 and 1.
'''
def predict_probability(feature_matrix, coefficients):
    # Take dot product of feature_matrix and coefficients  
    # YOUR CODE HERE
    score = np.dot(feature_matrix, coefficients)
    
    # Compute P(y_i = +1 | x_i, w) using the link function
    # YOUR CODE HERE
    predictions = 1/ (1 + np.exp(-score))
    
    # return predictions
    return predictions
```

**Aside**. How the link function works with matrix algebra

Since the word counts are stored as columns in **feature_matrix**, each $i$-th row of the matrix corresponds to the feature vector $h(\mathbf{x}_i)$:
$$
[\text{feature_matrix}] =
\left[
\begin{array}{c}
h(\mathbf{x}_1)^T \\
h(\mathbf{x}_2)^T \\
\vdots \\
h(\mathbf{x}_N)^T
\end{array}
\right] =
\left[
\begin{array}{cccc}
h_0(\mathbf{x}_1) & h_1(\mathbf{x}_1) & \cdots & h_D(\mathbf{x}_1) \\
h_0(\mathbf{x}_2) & h_1(\mathbf{x}_2) & \cdots & h_D(\mathbf{x}_2) \\
\vdots & \vdots & \ddots & \vdots \\
h_0(\mathbf{x}_N) & h_1(\mathbf{x}_N) & \cdots & h_D(\mathbf{x}_N)
\end{array}
\right]
$$

By the rules of matrix multiplication, the score vector containing elements $\mathbf{w}^T h(\mathbf{x}_i)$ is obtained by multiplying **feature_matrix** and the coefficient vector $\mathbf{w}$.
$$
[\text{score}] =
[\text{feature_matrix}]\mathbf{w} =
\left[
\begin{array}{c}
h(\mathbf{x}_1)^T \\
h(\mathbf{x}_2)^T \\
\vdots \\
h(\mathbf{x}_N)^T
\end{array}
\right]
\mathbf{w}
= \left[
\begin{array}{c}
h(\mathbf{x}_1)^T\mathbf{w} \\
h(\mathbf{x}_2)^T\mathbf{w} \\
\vdots \\
h(\mathbf{x}_N)^T\mathbf{w}
\end{array}
\right]
= \left[
\begin{array}{c}
\mathbf{w}^T h(\mathbf{x}_1) \\
\mathbf{w}^T h(\mathbf{x}_2) \\
\vdots \\
\mathbf{w}^T h(\mathbf{x}_N)
\end{array}
\right]
$$

**Checkpoint**

Just to make sure you are on the right track, we have provided a few examples. If your `predict_probability` function is implemented correctly, then the outputs will match:


```python
dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])

correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),          1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_predictions = np.array( [ 1./(1+np.exp(-correct_scores[0])), 1./(1+np.exp(-correct_scores[1])) ] )

print ('The following outputs must match ')
print ('------------------------------------------------')
print ('correct_predictions           =', correct_predictions)
print ('output of predict_probability =', predict_probability(dummy_feature_matrix, dummy_coefficients))
```

    The following outputs must match 
    ------------------------------------------------
    correct_predictions           = [0.98201379 0.26894142]
    output of predict_probability = [0.98201379 0.26894142]
    

## Compute derivative of log likelihood with respect to a single coefficient

Recall from lecture:
$$
\frac{\partial\ell}{\partial w_j} = \sum_{i=1}^N h_j(\mathbf{x}_i)\left(\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w})\right)
$$

We will now write a function that computes the derivative of log likelihood with respect to a single coefficient $w_j$. The function accepts two arguments:
* `errors` vector containing $\mathbf{1}[y_i = +1] - P(y_i = +1 | \mathbf{x}_i, \mathbf{w})$ for all $i$.
* `feature` vector containing $h_j(\mathbf{x}_i)$  for all $i$. 

Complete the following code block:


```python
def feature_derivative(errors, feature):     
    # Compute the dot product of errors and feature
    derivative = np.dot(errors, feature)
    
    # Return the derivative
    return derivative
```

In the main lecture, our focus was on the likelihood.  In the advanced optional video, however, we introduced a transformation of this likelihood---called the log likelihood---that simplifies the derivation of the gradient and is more numerically stable.  Due to its numerical stability, we will use the log likelihood instead of the likelihood to assess the algorithm.

The log likelihood is computed using the following formula (see the advanced optional video if you are curious about the derivation of this equation):

$$\ell\ell(\mathbf{w}) = \sum_{i=1}^N \Big( (\mathbf{1}[y_i = +1] - 1)\mathbf{w}^T h(\mathbf{x}_i) - \ln\left(1 + \exp(-\mathbf{w}^T h(\mathbf{x}_i))\right) \Big) $$

We provide a function to compute the log likelihood for the entire dataset. 


```python
def compute_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    
    # Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]
    
    lp = np.sum((indicator-1)*scores - logexp)
    return lp
```

**Checkpoint**

Just to make sure we are on the same page, run the following code block and check that the outputs match.


```python
dummy_feature_matrix = np.array([[1.,2.,3.], [1.,-1.,-1]])
dummy_coefficients = np.array([1., 3., -1.])
dummy_sentiment = np.array([-1, 1])

correct_indicators  = np.array( [ -1==+1,                                       1==+1 ] )
correct_scores      = np.array( [ 1.*1. + 2.*3. + 3.*(-1.),                     1.*1. + (-1.)*3. + (-1.)*(-1.) ] )
correct_first_term  = np.array( [ (correct_indicators[0]-1)*correct_scores[0],  (correct_indicators[1]-1)*correct_scores[1] ] )
correct_second_term = np.array( [ np.log(1. + np.exp(-correct_scores[0])),      np.log(1. + np.exp(-correct_scores[1])) ] )

correct_ll          =      sum( [ correct_first_term[0]-correct_second_term[0], correct_first_term[1]-correct_second_term[1] ] ) 

print ('The following outputs must match ')
print ('------------------------------------------------')
print ('correct_log_likelihood           =', correct_ll)
print ('output of compute_log_likelihood =', compute_log_likelihood(dummy_feature_matrix, dummy_sentiment, dummy_coefficients))
```

    The following outputs must match 
    ------------------------------------------------
    correct_log_likelihood           = -5.331411615436032
    output of compute_log_likelihood = -5.331411615436032
    

## Taking gradient steps

Now we are ready to implement our own logistic regression. All we have to do is to write a gradient ascent function that takes gradient steps towards the optimum. 

Complete the following function to solve the logistic regression model using gradient ascent:


```python
from math import sqrt

def logistic_regression(feature_matrix, sentiment, initial_coefficients, step_size, max_iter):
    coefficients = np.array(initial_coefficients) # make sure it's a numpy array
    for itr in range(max_iter):

        # Predict P(y_i = +1|x_i,w) using your predict_probability() function
        # YOUR CODE HERE
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in range(len(coefficients)): # loop over each coefficient
            
            # Recall that feature_matrix[:,j] is the feature column associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a variable called derivative
            # YOUR CODE HERE
            derivative = feature_derivative(errors, feature_matrix.iloc[:,j])
            
            # add the step size times the derivative to the current coefficient
            ## YOUR CODE HERE
            coefficients[j] += step_size * derivative
        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood(feature_matrix, sentiment, coefficients)
            print('iteration is ' + str(itr))
            print('log likelyhood is ' + str(lp))
#             print ('iteration %*d: log likelihood of observed labels = %.8f') % \
#                 (int(np.ceil(np.log10(max_iter))), itr, lp)
    return coefficients
```

Now, let us run the logistic regression solver.


```python
coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(194),
                                   step_size=1e-7, max_iter=301)
```

    iteration is 0
    log likelyhood is -36780.91768478126
    iteration is 1
    log likelyhood is -36775.1343471232
    iteration is 2
    log likelyhood is -36769.3571356369
    iteration is 3
    log likelyhood is -36763.5860323965
    iteration is 4
    log likelyhood is -36757.82101961526
    iteration is 5
    log likelyhood is -36752.06207964397
    iteration is 6
    log likelyhood is -36746.30919496959
    iteration is 7
    log likelyhood is -36740.562348213745
    iteration is 8
    log likelyhood is -36734.82152213133
    iteration is 9
    log likelyhood is -36729.08669960912
    iteration is 10
    log likelyhood is -36723.357863664314
    iteration is 11
    log likelyhood is -36717.634997443216
    iteration is 12
    log likelyhood is -36711.91808421987
    iteration is 13
    log likelyhood is -36706.20710739464
    iteration is 14
    log likelyhood is -36700.502050493
    iteration is 15
    log likelyhood is -36694.8028971641
    iteration is 20
    log likelyhood is -36666.39512032845
    iteration is 30
    log likelyhood is -36610.01327118031
    iteration is 40
    log likelyhood is -36554.19728365376
    iteration is 50
    log likelyhood is -36498.93316099372
    iteration is 60
    log likelyhood is -36444.207839138624
    iteration is 70
    log likelyhood is -36390.009094487075
    iteration is 80
    log likelyhood is -36336.3254614401
    iteration is 90
    log likelyhood is -36283.14615870863
    iteration is 100
    log likelyhood is -36230.46102346926
    iteration is 200
    log likelyhood is -35728.89418769386
    iteration is 300
    log likelyhood is -35268.51212682766
    


```python
# coefficients = logistic_regression(feature_matrix, sentiment, initial_coefficients=np.zeros(193),
#                                    step_size=1e-7, max_iter=301)
```

**Quiz Question:** As each iteration of gradient ascent passes, does the log likelihood increase or decrease?

## Predicting sentiments

Recall from lecture that class predictions for a data point $\mathbf{x}$ can be computed from the coefficients $\mathbf{w}$ using the following formula:
$$
\hat{y}_i = 
\left\{
\begin{array}{ll}
      +1 & \mathbf{x}_i^T\mathbf{w} > 0 \\
      -1 & \mathbf{x}_i^T\mathbf{w} \leq 0 \\
\end{array} 
\right.
$$

Now, we will write some code to compute class predictions. We will do this in two steps:
* **Step 1**: First compute the **scores** using **feature_matrix** and **coefficients** using a dot product.
* **Step 2**: Using the formula above, compute the class predictions from the scores.

Step 1 can be implemented as follows:


```python
# Compute the scores as a dot product between feature_matrix and coefficients.
scores = np.dot(feature_matrix, coefficients)
```


```python
products['scores'] = scores
```

Now, complete the following code block for **Step 2** to compute the class predictions using the **scores** obtained above:


```python
products['sentiment_prediction'] = products['scores'].apply(lambda x: +1 if x>0  else -1)
```


```python
products
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
      <th>sentiment</th>
      <th>review_clean</th>
      <th>baby</th>
      <th>one</th>
      <th>great</th>
      <th>love</th>
      <th>use</th>
      <th>...</th>
      <th>wish</th>
      <th>buying</th>
      <th>babies</th>
      <th>won</th>
      <th>tub</th>
      <th>almost</th>
      <th>either</th>
      <th>contains_perfect</th>
      <th>scores</th>
      <th>sentiment_prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Stop Pacifier Sucking without tears with Thumb...</td>
      <td>All of my kids have cried non-stop when I trie...</td>
      <td>5</td>
      <td>1</td>
      <td>All of my kids have cried nonstop when I tried...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.051046</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>We wanted to get something to keep track of ou...</td>
      <td>5</td>
      <td>1</td>
      <td>We wanted to get something to keep track of ou...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.029365</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nature's Lullabies Second Year Sticker Calendar</td>
      <td>My daughter had her 1st baby over a year ago. ...</td>
      <td>5</td>
      <td>1</td>
      <td>My daughter had her 1st baby over a year ago S...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.024116</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lamaze Peekaboo, I Love You</td>
      <td>One of baby's first and favorite books, and it...</td>
      <td>4</td>
      <td>1</td>
      <td>One of babys first and favorite books and it i...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.007869</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SoftPlay Peek-A-Boo Where's Elmo A Children's ...</td>
      <td>Very cute interactive book! My son loves this ...</td>
      <td>5</td>
      <td>1</td>
      <td>Very cute interactive book My son loves this b...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.131819</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Our Baby Girl Memory Book</td>
      <td>Beautiful book, I love it to record cherished ...</td>
      <td>5</td>
      <td>1</td>
      <td>Beautiful book I love it to record cherished t...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.129835</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hunnt&amp;reg; Falling Flowers and Birds Kids Nurs...</td>
      <td>Try this out for a spring project !Easy ,fun a...</td>
      <td>5</td>
      <td>1</td>
      <td>Try this out for a spring project Easy fun and...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.008035</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Blessed By Pope Benedict XVI Divine Mercy Full...</td>
      <td>very nice Divine Mercy Pendant of Jesus now on...</td>
      <td>5</td>
      <td>1</td>
      <td>very nice Divine Mercy Pendant of Jesus now on...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.039765</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Cloth Diaper Pins Stainless Steel Traditional ...</td>
      <td>We bought the pins as my 6 year old Autistic s...</td>
      <td>4</td>
      <td>1</td>
      <td>We bought the pins as my 6 year old Autistic s...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.007329</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Cloth Diaper Pins Stainless Steel Traditional ...</td>
      <td>It has been many years since we needed diaper ...</td>
      <td>5</td>
      <td>1</td>
      <td>It has been many years since we needed diaper ...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.052315</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Newborn Baby Tracker&amp;reg; - Round the Clock Ch...</td>
      <td>We found this book at a rummage sale and found...</td>
      <td>5</td>
      <td>1</td>
      <td>We found this book at a rummage sale and found...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-0.027536</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Newborn Baby Tracker&amp;reg; - Round the Clock Ch...</td>
      <td>I'm a new mom and I was looking for something ...</td>
      <td>5</td>
      <td>1</td>
      <td>Im a new mom and I was looking for something t...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.082426</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Newborn Baby Tracker&amp;reg; - Round the Clock Ch...</td>
      <td>I loved how this book was set up to keep track...</td>
      <td>5</td>
      <td>1</td>
      <td>I loved how this book was set up to keep track...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.021704</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Newborn Baby Tracker&amp;reg; - Round the Clock Ch...</td>
      <td>I received this at my baby shower and it has b...</td>
      <td>5</td>
      <td>1</td>
      <td>I received this at my baby shower and it has b...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.124914</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Neurosmith - Music Blocks with Mozart Music Ca...</td>
      <td>My daughter started playing with her Music Blo...</td>
      <td>5</td>
      <td>1</td>
      <td>My daughter started playing with her Music Blo...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.187858</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Neurosmith - Music Blocks with Mozart Music Ca...</td>
      <td>It takes a youthful spirit of inquiry and fasc...</td>
      <td>5</td>
      <td>1</td>
      <td>It takes a youthful spirit of inquiry and fasc...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.043931</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Neurosmith - Music Blocks with Mozart Music Ca...</td>
      <td>This is an interesting and educational toy.  I...</td>
      <td>4</td>
      <td>1</td>
      <td>This is an interesting and educational toy  I ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.093460</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Neurosmith - Music Blocks with Mozart Music Ca...</td>
      <td>Recently I have purchased the musical mozart b...</td>
      <td>5</td>
      <td>1</td>
      <td>Recently I have purchased the musical mozart b...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.071406</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>my first fish bowl by lamaze / learning curve</td>
      <td>We first bought this toy for our oldest child ...</td>
      <td>5</td>
      <td>1</td>
      <td>We first bought this toy for our oldest child ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.044850</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Pedal Farm Tractor</td>
      <td>We bought this tractor for our 2 and a half-ye...</td>
      <td>5</td>
      <td>1</td>
      <td>We bought this tractor for our 2 and a halfyea...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.053937</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>This is a great toy.  The wheels really work a...</td>
      <td>5</td>
      <td>1</td>
      <td>This is a great toy  The wheels really work an...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.082627</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>After being inundated with toys that require b...</td>
      <td>5</td>
      <td>1</td>
      <td>After being inundated with toys that require b...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.208033</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>We bought these trucks for our 15 mo. old daug...</td>
      <td>5</td>
      <td>1</td>
      <td>We bought these trucks for our 15 mo old daugh...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.145432</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>For well over a year my son has enjoyed stacki...</td>
      <td>5</td>
      <td>1</td>
      <td>For well over a year my son has enjoyed stacki...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.057522</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>We just bought this for my 9 month old daughte...</td>
      <td>5</td>
      <td>1</td>
      <td>We just bought this for my 9 month old daughte...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.244022</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Fisher Price Nesting Action Vehicles</td>
      <td>This is a wonderful toy that is fun, education...</td>
      <td>5</td>
      <td>1</td>
      <td>This is a wonderful toy that is fun educationa...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.220524</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Sassy Who Loves Baby? Photo Album Book with te...</td>
      <td>I bought this for a new granddaughter.  I will...</td>
      <td>5</td>
      <td>1</td>
      <td>I bought this for a new granddaughter  I will ...</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.121861</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Earlyears: Earl E. Bird with Teething Rings</td>
      <td>We received an Earl E. Bird as a gift when we ...</td>
      <td>5</td>
      <td>1</td>
      <td>We received an Earl E Bird as a gift when we h...</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.253597</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Earlyears: Earl E. Bird with Teething Rings</td>
      <td>This little toy is safe for infants, and offer...</td>
      <td>5</td>
      <td>1</td>
      <td>This little toy is safe for infants and offers...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.137596</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>My Quiet Book, Fabric Activity Book for Children</td>
      <td>This is exactly like the one I had when I was ...</td>
      <td>5</td>
      <td>1</td>
      <td>This is exactly like the one I had when I was ...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.005373</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>53042</th>
      <td>Beautiful Luxury Nightcap Baby Suspender Baby ...</td>
      <td>I only put 1 star because I had to. The photos...</td>
      <td>1</td>
      <td>-1</td>
      <td>I only put 1 star because I had to The photos ...</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.199438</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53043</th>
      <td>Beautiful Luxury Nightcap Baby Suspender Baby ...</td>
      <td>Amazon, please take this down. The pictures sh...</td>
      <td>1</td>
      <td>-1</td>
      <td>Amazon please take this down The pictures show...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.012100</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53044</th>
      <td>Blackcell 2pairs Infant Toddler Baby Knee Pad ...</td>
      <td>Not very impressed. They are cute, however the...</td>
      <td>1</td>
      <td>-1</td>
      <td>Not very impressed They are cute however they ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.016683</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53045</th>
      <td>Nuby Soft Spout Easy Grip Sippy Cup - 4 pk.</td>
      <td>I agree with the other 2 posters. These sippy ...</td>
      <td>1</td>
      <td>-1</td>
      <td>I agree with the other 2 posters These sippy c...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.032601</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53046</th>
      <td>VTech Communications Safe and Sound Digital Au...</td>
      <td>These monitors do not work at all, I even atte...</td>
      <td>1</td>
      <td>-1</td>
      <td>These monitors do not work at all I even attem...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.205256</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53047</th>
      <td>Giftoyou(TM) High Quality Lightweight Retracta...</td>
      <td>Our family purchased numerous of these because...</td>
      <td>1</td>
      <td>-1</td>
      <td>Our family purchased numerous of these because...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.155067</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53048</th>
      <td>Giftoyou(TM) High Quality Lightweight Retracta...</td>
      <td>I was so happy when i bought this charger for ...</td>
      <td>1</td>
      <td>-1</td>
      <td>I was so happy when i bought this charger for ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.052437</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53049</th>
      <td>zTcase&amp;trade; Bluetooth Wireless Keyboard Case...</td>
      <td>I was very excited to receive this keyboard ca...</td>
      <td>1</td>
      <td>-1</td>
      <td>I was very excited to receive this keyboard ca...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.150105</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53050</th>
      <td>Geleeo Universal Stroller Cooling Pad - Pink</td>
      <td>I agree with the reviewer below, this pad you ...</td>
      <td>2</td>
      <td>-1</td>
      <td>I agree with the reviewer below this pad you w...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.000552</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53051</th>
      <td>4Moms mamaRoo Plush Infant Seat - Silver</td>
      <td>As a designer, I loved the look of the Mommaro...</td>
      <td>1</td>
      <td>-1</td>
      <td>As a designer I loved the look of the Mommaroo...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.053530</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53052</th>
      <td>4Moms mamaRoo Plush Infant Seat - Silver</td>
      <td>A friend told us to get this, said it was a &amp;#...</td>
      <td>2</td>
      <td>-1</td>
      <td>A friend told us to get this said it was a 34m...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.093022</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53053</th>
      <td>Merry Muscles Ergonomic Jumper Exerciser Baby ...</td>
      <td>once in this thing, my 2mo. son loves this... ...</td>
      <td>2</td>
      <td>-1</td>
      <td>once in this thing my 2mo son loves this howev...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.142628</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53054</th>
      <td>[Different Colors Available] Newisland&amp;reg; 79...</td>
      <td>I received this product free for review but de...</td>
      <td>2</td>
      <td>-1</td>
      <td>I received this product free for review but de...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-0.156573</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53055</th>
      <td>Vandot for Samsung Galaxy S4 / I9500 ULTRA SLI...</td>
      <td>Omg this case was so ugly and so hard that I t...</td>
      <td>1</td>
      <td>-1</td>
      <td>Omg this case was so ugly and so hard that I t...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.242001</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53056</th>
      <td>Nicerocker big Cute Sweet Kids Baby Girls Love...</td>
      <td>I only ordered one of this item and received 2...</td>
      <td>1</td>
      <td>-1</td>
      <td>I only ordered one of this item and received 2...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.163957</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53057</th>
      <td>Clevamama 3-in-1 Sleep, Sit and Play Travel Ma...</td>
      <td>The mattress is supposedly hypoallergenic clev...</td>
      <td>1</td>
      <td>-1</td>
      <td>The mattress is supposedly hypoallergenic clev...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.173353</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53058</th>
      <td>Graco Argos 65 3-in-1 Harness Booster, Link</td>
      <td>Short story, I was very disappointed with the ...</td>
      <td>1</td>
      <td>-1</td>
      <td>Short story I was very disappointed with the q...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.485708</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53059</th>
      <td>K&amp;amp;C Baby Bath Seat Support Sling Shower Me...</td>
      <td>Absolute rip off!!! Not impressed at all this ...</td>
      <td>1</td>
      <td>-1</td>
      <td>Absolute rip off Not impressed at all this was...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.045782</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53060</th>
      <td>Aqueduck Faucet Extender &amp;amp; Handle Extended...</td>
      <td>I wish I had bought the faucet extender and ha...</td>
      <td>2</td>
      <td>-1</td>
      <td>I wish I had bought the faucet extender and ha...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.075543</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53061</th>
      <td>ANDI ROSE Baby Toddlers Floral Printed Ruffle ...</td>
      <td>Definitely made extremely poorly in china. Is ...</td>
      <td>1</td>
      <td>-1</td>
      <td>Definitely made extremely poorly in china Is n...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.037075</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53062</th>
      <td>Vandot 2 in1 Accessory Set 3D Leather Case Lit...</td>
      <td>Cute but cheaply made.. The part where you put...</td>
      <td>2</td>
      <td>-1</td>
      <td>Cute but cheaply made The part where you put y...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.078109</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53063</th>
      <td>Umai Authentic Hazelwood and CHERRY RAW (Unpol...</td>
      <td>Made no difference :/</td>
      <td>1</td>
      <td>-1</td>
      <td>Made no difference</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.005162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53064</th>
      <td>360 Degree Rotating Cover Case for Samsung Gal...</td>
      <td>Be careful it stains your screen protector.  B...</td>
      <td>1</td>
      <td>-1</td>
      <td>Be careful it stains your screen protector  Bo...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.064949</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53065</th>
      <td>Summer Infant Pop 'n Play Portable Playard</td>
      <td>Good idea but too dangerous. I really wanted t...</td>
      <td>2</td>
      <td>-1</td>
      <td>Good idea but too dangerous I really wanted to...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.018230</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53066</th>
      <td>Freeens Cool Seat Liner Breathing with 3d Mesh...</td>
      <td>It doesn't stay input. My daughter was sliding...</td>
      <td>1</td>
      <td>-1</td>
      <td>It doesnt stay input My daughter was sliding o...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.027149</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53067</th>
      <td>Samsung Baby Care Washer, Stainless Platinum, ...</td>
      <td>My infant goes to a really crappy daycare, and...</td>
      <td>1</td>
      <td>-1</td>
      <td>My infant goes to a really crappy daycare and ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.010504</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53068</th>
      <td>Mud Pie Milestone Stickers, Boy</td>
      <td>Pretty please open and inspect these stickers ...</td>
      <td>1</td>
      <td>-1</td>
      <td>Pretty please open and inspect these stickers ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.005606</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53069</th>
      <td>Best BIB for Baby - Soft Bib (Pink-Elephant)</td>
      <td>Great 5-Star Product but An Obvious knock-off ...</td>
      <td>1</td>
      <td>-1</td>
      <td>Great 5Star Product but An Obvious knockoff of...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.409863</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>53070</th>
      <td>Bouncy&amp;reg; Inflatable Real Feel Hopping Cow</td>
      <td>When I received the item my initial thought wa...</td>
      <td>2</td>
      <td>-1</td>
      <td>When I received the item my initial thought wa...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.014114</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53071</th>
      <td>Maxboost iPhone 5S/5 Case - Protective Snap-on...</td>
      <td>I got this case in the mail today, it came on ...</td>
      <td>2</td>
      <td>-1</td>
      <td>I got this case in the mail today it came on t...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0.067559</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>53072 rows × 201 columns</p>
</div>



** Quiz Question: ** How many reviews were predicted to have positive sentiment?


```python
products['sentiment_prediction'].value_counts()
```




    -1    27946
     1    25126
    Name: sentiment_prediction, dtype: int64



## Measuring accuracy

We will now measure the classification accuracy of the model. Recall from the lecture that the classification accuracy can be computed as follows:

$$
\mbox{accuracy} = \frac{\mbox{# correctly classified data points}}{\mbox{# total data points}}
$$

Complete the following code block to compute the accuracy of the model.


```python
# def model_accuracy(model, input_matrix, target_parameter):
#     recall = model.predict(input_matrix)
#     TF_array = (recall == target_parameter)
#     correct_counts = np.count_nonzero(TF_array == True)
#     accu_math = correct_counts/len(TF_array)
#     return accu_math
```


```python
compare_results = (products['sentiment_prediction']==products['sentiment']).value_counts()
```


```python
compare_results
```




    True     39903
    False    13169
    dtype: int64




```python
type(compare_results)
```




    pandas.core.series.Series




```python
type(compare_results[False])
```




    numpy.int64




```python
num_mistakes = compare_results[False]
accuracy = (len(products)-num_mistakes)/len(products)
# print(accuracy)
print ("-----------------------------------------------------")
print ('# Reviews   correctly classified =', len(products) - num_mistakes)
print ('# Reviews incorrectly classified =', num_mistakes)
print ('# Reviews total                  =', len(products))
print ("-----------------------------------------------------")
print ('Accuracy = {:.2%}'.format(accuracy))
```

    -----------------------------------------------------
    # Reviews   correctly classified = 39903
    # Reviews incorrectly classified = 13169
    # Reviews total                  = 53072
    -----------------------------------------------------
    Accuracy = 75.19%
    

**Quiz Question**: What is the accuracy of the model on predictions made above? (round to 2 digits of accuracy)

## Which words contribute most to positive & negative sentiments?

Recall that in Module 2 assignment, we were able to compute the "**most positive words**". These are words that correspond most strongly with positive reviews. In order to do this, we will first do the following:
* Treat each coefficient as a tuple, i.e. (**word**, **coefficient_value**).
* Sort all the (**word**, **coefficient_value**) tuples by **coefficient_value** in descending order.


```python
len(coefficients)
```




    194




```python
coefficients
```




    array([ 5.16220157e-03,  1.55656966e-02, -8.50204675e-03,  6.65460842e-02,
            6.58907629e-02,  5.01743882e-03, -5.38601484e-02, -3.50488413e-03,
            6.47945868e-02,  4.54356263e-02,  3.98353364e-03,  2.00775410e-02,
            3.01350011e-02, -2.87115530e-02,  1.52161964e-02,  2.72592062e-04,
            1.19448177e-02, -1.82461935e-02, -1.21706420e-02, -4.15110334e-02,
            2.76820391e-03,  1.77031999e-02, -4.39700067e-03,  4.49764014e-02,
            9.90916464e-03,  8.99239081e-04, -1.36219516e-03,  1.26859357e-02,
            8.26466695e-03, -2.77426972e-02,  6.10128809e-04,  1.54084501e-02,
           -1.32134753e-02, -3.00512492e-02,  2.97399371e-02,  1.84087080e-02,
            2.86178752e-03, -1.05768015e-02, -6.57350362e-04, -1.01476555e-02,
           -4.79579528e-03,  7.50891810e-03,  4.27938289e-03,  3.06785501e-03,
           -2.20317661e-03,  9.57273354e-03,  9.91666827e-05, -1.98462567e-02,
            1.75702722e-02,  1.55478612e-03, -1.77375440e-02,  9.78324102e-03,
            1.17031606e-02, -7.35345937e-03, -6.08714030e-03,  6.43766808e-03,
            1.07159665e-02, -3.05345476e-03,  7.17190727e-03,  5.73320003e-03,
            4.60661876e-03, -5.20588421e-03,  6.71012331e-03,  9.03281814e-03,
            1.74563147e-03,  6.00279979e-03,  1.20181744e-02, -1.83594607e-02,
           -6.91010811e-03, -1.38687273e-02, -1.50406590e-02,  5.92353611e-03,
            5.67478991e-03, -5.28786220e-03,  3.08147864e-03,  5.53751236e-03,
            1.49917916e-02, -3.35666000e-04, -3.30695153e-02, -4.78990943e-03,
           -6.41368859e-03,  7.99938935e-03, -8.61390444e-04,  1.68052959e-02,
            1.32539901e-02,  1.72307051e-03,  2.98030675e-03,  8.58284300e-03,
            1.17082481e-02,  2.80825907e-03,  2.18724016e-03,  1.68824711e-02,
           -4.65973741e-03,  1.51368285e-03, -1.09509122e-02,  9.17842898e-03,
           -1.88572281e-04, -3.89820373e-02, -2.44821005e-02, -1.87023714e-02,
           -2.13943485e-02, -1.29690465e-02, -1.71378670e-02, -1.37566767e-02,
           -1.49770449e-02, -5.10287978e-03, -2.89789761e-02, -1.48663194e-02,
           -1.28088380e-02, -1.07709355e-02, -6.95286915e-03, -5.04082164e-03,
           -9.25914404e-03, -2.40427481e-02, -2.65927785e-02, -1.97320937e-03,
           -5.04127508e-03, -7.00791912e-03, -3.48088523e-03, -6.40958916e-03,
           -4.07497010e-03, -6.30054296e-03, -1.09187932e-02, -1.26051900e-02,
           -1.66895314e-03, -7.76418781e-03, -5.15960485e-04, -1.94199551e-03,
           -1.24761586e-03, -5.01291731e-03, -9.12049191e-03, -7.22098801e-03,
           -8.31782981e-03, -5.60573348e-03, -1.47098335e-02, -9.31520819e-03,
           -2.22034402e-03, -7.07573098e-03, -5.10115608e-03, -8.93572862e-03,
           -1.27545713e-02, -7.04171991e-03, -9.76219676e-04,  4.12091713e-04,
            8.29251160e-04,  2.64661064e-03, -7.73228782e-03,  1.53471164e-03,
           -7.37263060e-03, -3.73694386e-03, -3.81416409e-03, -1.64575145e-03,
           -3.31887732e-03,  1.22257832e-03,  1.36699286e-05, -3.01866601e-03,
           -1.02826343e-02, -1.06691327e-02,  2.23639046e-03, -9.87424798e-03,
           -1.02192048e-02, -3.41330929e-03,  3.34489960e-03, -3.50984516e-03,
           -6.26283150e-03, -7.22419943e-03, -5.47016154e-03, -1.25063947e-02,
           -2.47805699e-03, -1.60017985e-02, -6.40098934e-03, -4.26644386e-03,
           -1.55376990e-02,  2.31349237e-03, -9.06653337e-03, -6.30012672e-03,
           -1.21010303e-02, -3.02578875e-03, -6.76289718e-03, -5.65498722e-03,
           -6.87050239e-03, -1.18950595e-02, -1.86489236e-04, -1.15230476e-02,
            2.81533219e-03, -8.10150295e-03, -1.00062131e-02,  4.02037651e-03,
           -5.44300346e-03,  2.85818985e-03,  1.19885003e-04, -6.47587687e-03,
           -1.14493516e-03, -7.09205934e-03])




```python
coefficients = list(coefficients[1:]) # exclude intercept
word_coefficient_tuples = [(word, coefficient) for word, coefficient in zip(important_words, coefficients)]
word_coefficient_tuples = sorted(word_coefficient_tuples, key=lambda x:x[1], reverse=True)
```

Now, **word_coefficient_tuples** contains a sorted list of (**word**, **coefficient_value**) tuples. The first 10 elements in this list correspond to the words that are most positive.

### Ten "most positive" words

Now, we compute the 10 words that have the most positive coefficient values. These words are associated with positive sentiment.


```python
word_coefficient_tuples[0:10]
```




    [('great', 0.06654608417045771),
     ('love', 0.06589076292212327),
     ('easy', 0.0647945868025784),
     ('little', 0.045435626308421365),
     ('loves', 0.044976401394906045),
     ('well', 0.03013500109210708),
     ('perfect', 0.029739937104968462),
     ('old', 0.020077541034775374),
     ('nice', 0.018408707995268992),
     ('daughter', 0.017703199905701694)]



** Quiz Question:** Which word is **not** present in the top 10 "most positive" words?

### Ten "most negative" words

Next, we repeat this exercise on the 10 most negative words.  That is, we compute the 10 words that have the most negative coefficient values. These words are associated with negative sentiment.


```python
word_coefficient_tuples[-10:]
```




    [('monitor', -0.02448210054589172),
     ('return', -0.026592778462247283),
     ('back', -0.027742697230661337),
     ('get', -0.028711552980192578),
     ('disappointed', -0.02897897614231707),
     ('even', -0.030051249236035808),
     ('work', -0.03306951529475273),
     ('money', -0.038982037286487116),
     ('product', -0.04151103339210888),
     ('would', -0.053860148445203135)]



** Quiz Question:** Which word is **not** present in the top 10 "most negative" words?
