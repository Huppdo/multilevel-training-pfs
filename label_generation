import psycopg2
import pandas as pd
import re
import html
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import warnings
from bs4 import MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper
from joblib import dump, load
from sklearn import svm
import os.path

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Replace these with your database credentials
host = ""
dbname = ""
user = ""
password = ""

def clean_db_text(text, issues):
    """
    Cleans the text in the db by removing bad characters

    Parameters:
    text (str): Text to be cleaned.
    issues (list): Issues with db text

    Returns:
    text (str): Cleaned text.

    """
    if not isinstance(text, str):
        return text
    try:
        text = text.encode("utf-8", "ignore").decode("utf-8")
    except:
        issues['encoding'] += 1

    text = html.unescape(text)

    text = re.sub(r'<[^>]+>', '', text)
    illegal_characters = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
    match = illegal_characters.search(text)
    if match:
        issues['illegal_characters'] += 1

    text = illegal_characters.sub("", text)
    return text

def collect_db_unique_letters(json_filename, docket_id):
    """
    Collects the unique letters from the db and saves it into a json file

    Parameters:
    json_filename (String): The filename of the JSON file.
    docket_id (String): The docket ID to be used

    Returns:
    df (DataFrame): A dataframe with all the letters in the docket
    """

    # Database connection
    connection = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)

    # Your SQL query
    sql_query = "SELECT DISTINCT combined.document_id, combined.attach_seq, combined.comment, " \
                "provisions.provision_name, provisions.sub_provision_name," \
                "metadata.major_commenter_group, metadata.commenter " \
                "FROM regulations_comments_dev.reg_comments_attachments_combined combined " \
                "JOIN regulations_comments_dev.reg_documents_metadata metadata ON combined.document_id = metadata.document_id " \
                "JOIN regulations_comments_dev.reg_provisions provisions ON combined.document_id = provisions.document_id AND " \
                "combined.attach_seq = provisions.attach_seq " \
                "WHERE combined.document_id IN ( SELECT document_id FROM comment_review.cr_comments " \
                "WHERE primary_review_status IN (4) AND qa_review_status IN (4) AND docket_id = '"+docket_id+"' ) " \
                " AND provisions.provision_score >= 0.9 AND combined.docket_id = '"+docket_id+"' " \
                "AND metadata.major_commenter_group IN ('1', '2', '3') AND provisions.is_provision_accurate = TRUE;"

    # New Database Query
    # select core.tracking_number, core.attach_seq, core.final_document_id, core.comment,
    # topic, subtopic, commenter, mc_group_id
    # from core join topic_assignment on core.tracking_number = topic_assignment.tracking_number
    # and core.attach_seq = topic_assignment.attach_seq join commenter
    # on core.tracking_number = commenter.tracking_number
    # and core.attach_seq = commenter.attach_seq join state_tracking
    # on core.tracking_number = state_tracking.tracking_number
    # and core.attach_seq = state_tracking.attach_seq
    # where docket_id = 'CMS-2022-0113' and pr_status = qa_status and qa_status = 4 and score >= 0.9
    # and is_topic_enabled is True

    # Execute the query and fetch the results
    with connection.cursor() as cursor:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results, columns=column_names)

    issues = {'encoding': 0, 'illegal_characters': 0}

    # Clean all columns
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: clean_db_text(x, issues) if isinstance(x, str) else x)

    # Print the summary of issues
    #print(f"Encoding issues found: {issues['encoding']}")
    #print(f"Illegal character issues found: {issues['illegal_characters']}")

    # Save the DataFrame as a JSON file
    if json_filename != '':
        df.to_json(json_filename, orient="records", lines=True)

    # Close the connection
    connection.close()
    return df

def collect_docket_topics(docket_id):
    """
    Collects docket topics from a specific docket

    Parameters:
    docket_id (String):  The docket ID to be used

    Returns:
    topics_list (List): A list of topics 
    """

    # Database connection
    connection = psycopg2.connect(host=host, dbname=dbname, user=user, password=password)

    # Your SQL query
    sql_query = "SELECT DISTINCT topic FROM comment_review.cr_topics WHERE docket_id = '"+docket_id+"';"

    # Execute the query and fetch the results
    with connection.cursor() as cursor:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results, columns=column_names)

    # list of values of 'topic' column
    topics_list = df['topic'].tolist()

    # Close the connection
    connection.close()
    return topics_list

def cleaner(text):
    """
    Cleans the text by removing html tags, special characters and stopwords.

    Parameters:
    text (String): Text to be cleaned.

    Returns:
    text (String): Cleaned text.
    """
    patterns = r'(&[rl]dquo;|&[rl]squo;|&nbsp;|<br\/>|<br>|<br\s*\/>|&amp;)'
    text = re.sub(patterns, ' ', text)
    text = html.unescape(text)
    text = BeautifulSoup(text, 'html.parser').get_text(separator=' ')
    text = re.sub(' +', ' ', text).strip()
    words = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text.lower()) if word not in stop_words]
    words = [re.sub(r'(?<=\s)\((?=\s)|(?<=\s)\)(?=\s)', '', word) for word in words]
    return ' '.join(words)

def clean_comments(df):
    """
    Applies the cleaner function to the comment column of the dataframe.

    Parameters:
    df (DataFrame): Dataframe with a 'comment' column.

    Returns:
    df (DataFrame): Dataframe with cleaned 'comment' column.
    """
    try:
        df['comment'] = df['comment'].apply(cleaner)
    except Exception as e:
        print(f"An error occurred while cleaning comments: {e}")
        raise e
    return df

def partial_pipe_fit(pipeline_obj, X, Y):
    pipeline_obj.named_steps['tfidf'].fit(X, Y)
    pipeline_obj.named_steps['model'].partial_fit(X,Y)

def train_and_save_model(df, mlb_provision):
    """
    Trains the logistic regression model and saves it.

    Parameters:
    df (DataFrame): Dataframe with 'comment' column as features and 'provision_name' as target.
    mlb_provision (MultiLabelBinarizer): MultiLabelBinarizer instance.

    Returns:
    None
    """

    try:

        y_provision_encoded = mlb_provision.fit_transform(df['provision_name'])
        logreg_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=17, ngram_range=(1, 4), lowercase=True, stop_words='english')),
            ('model', OneVsRestClassifier(LogisticRegression(solver='saga', penalty='l1', C=100, max_iter=10000, random_state=0, n_jobs=-1)))
        ])
        logreg_pipeline.fit(df['comment'], y_provision_encoded)
        dump(logreg_pipeline, 'logreg_pipeline.joblib')
    except Exception as e:
        print(f"An error occurred while training the model: {e}")
        raise e

def predict_with_model(df, mlb_provision):
    """
    Predicts using the trained model and adds the predictions to the dataframe as a new column.

    Parameters:
    df (DataFrame): Dataframe with 'comment' column.
    mlb_provision (MultiLabelBinarizer): MultiLabelBinarizer instance.

    Returns:
    df (DataFrame): Dataframe with new column 'pred_provision_name' containing predictions.
    """
    try:
        logreg_pipeline = load('logreg_pipeline.joblib')
        df['pred_provision_name'] = mlb_provision.inverse_transform(logreg_pipeline.predict(df['comment']))
    except Exception as e:
        print(f"An error occurred while making predictions: {e}")
        raise e
    return df

def predict_comment_with_model(comment, mlb_provision):
    """
    Predicts using the trained model and adds the predictions to the dataframe as a new column.

    Parameters:
    df (DataFrame): Dataframe with 'comment' column.
    mlb_provision (MultiLabelBinarizer): MultiLabelBinarizer instance.

    Returns:
    df (DataFrame): Dataframe with new column 'pred_provision_name' containing predictions.
    """
    returnValue = None
    try:
        logreg_pipeline = load('logreg_pipeline.joblib')
        returnValue = mlb_provision.inverse_transform(logreg_pipeline.predict([cleaner(comment)]))
        #print(returnValue)
    except Exception as e:
        print(f"An error occurred while making predictions: {e}")
        raise e
    return returnValue[0]

def validate_model_results(df, mlb_provision):
    """
        Validates using provided data to create a model and test it.

        Parameters:
        df (DataFrame): Dataframe with 'comment' column.
        mlb_provision (MultiLabelBinarizer): MultiLabelBinarizer instance.

        Returns:
        tp (Int): True positives where topics found in both 'provision_name' and 'pred_provision_name' columns.
        fp (Int): False positives where topics found in 'pred_provision_name' column only.
        fn (Int): False negatives where topics found in 'provision_name' column only.
    """
    train_data = df.sample(frac = 0.8)
    test_data = df.drop(train_data.index)
    train_and_save_model(train_data,mlb_provision)
    test_data = predict_with_model(test_data,mlb_provision)
    tp=0;fp=0;fn=0
    for ind in test_data.index:
        tp = tp + len(np.intersect1d(test_data['provision_name'][ind], test_data['pred_provision_name'][ind]))
        fp = fp + len(np.setdiff1d(test_data['pred_provision_name'][ind], test_data['provision_name'][ind]))
        fn = fn + len(np.setdiff1d(test_data['provision_name'][ind], test_data['pred_provision_name'][ind]))
    return tp, fp, fn

def model_train(trainingCount, binarizer):
    """
    Main function to run all model training steps.

    Returns:
    None
    """

    print(f"Retraining model with {trainingCount} comments")

    # 1. Collect form letters from database
    # to collect from DB
    json_filename = "CMS-2022-0113_Unique-Letters.json"

    #Check if we already have the file (saves query time)
    if os.path.isfile(json_filename):
    # to load from a json file
        df = (pd.read_json(json_filename, lines=True)
                .astype({'comment': 'string'})
                .loc[lambda x: x['comment'].str.len() > 20]
                .reset_index(drop=True))
    else:
        df = collect_db_unique_letters(json_filename,'CMS-2022-0113')

    print("Collected unique comments from DB")

    # 2. Prepare comments
    df = df.groupby(['document_id', 'attach_seq']) \
        .agg({'comment': 'first', 'provision_name': lambda x: tuple(set(x)), 'sub_provision_name': lambda x: tuple(set(x)), 'major_commenter_group': 'first', 'commenter': 'first'}).reset_index()

    print("Prepared the comments for training")

    # Ensure provision_name is always a tuple
    df['provision_name'] = df['provision_name'].apply(lambda x: (x,) if isinstance(x, str) else x)

    # Clean comments
    df = clean_comments(df)

    print("Cleaned comments")

    train_and_save_model(df[1:trainingCount], binarizer)

    print(f"Model retrained with {trainingCount} comments")

def main():
    """
    Main function to clean data, train model, and make predictions.

    Returns:
    None
    """

    # 1. Collect form letters from database
    # to collect from DB
    json_filename = "CMS-2022-0113_Unique-Letters.json"
    df = collect_db_unique_letters(json_filename,'CMS-2022-0113')
    # to load from a json file
    df = (pd.read_json(json_filename, lines=True)
            .astype({'comment': 'string'})
            .loc[lambda x: x['comment'].str.len() > 20]
            .reset_index(drop=True))

    # 2. Prepare comments
    df = df.groupby(['document_id', 'attach_seq']) \
        .agg({'comment': 'first', 'provision_name': lambda x: tuple(set(x)), 'sub_provision_name': lambda x: tuple(set(x)), 'major_commenter_group': 'first', 'commenter': 'first'}).reset_index()

    # Ensure provision_name is always a tuple
    df['provision_name'] = df['provision_name'].apply(lambda x: (x,) if isinstance(x, str) else x)

    # Clean comments
    df = clean_comments(df)

    # 3. Collect and initialize a list of all provision names for this docket_id
    all_provision_names = collect_docket_topics('CMS-2022-0113')
    # Initialize the MultiLabelBinarizer with the classes
    mlb_provision = MultiLabelBinarizer(classes=all_provision_names)

    # 4. Train and save the model
    # automate by counting the finalized comment provisions like: 1000, 2000, etc.
    training_comment_count = 500
    train_and_save_model(df[1:training_comment_count], mlb_provision)

    # 5. Predict the non finalized comments with the model and save the predictions in a new dataframe column
    #df = predict_with_model(df, mlb_provision)
    aiTopics = predict_comment_with_model("Test 123", mlb_provision)
    print(aiTopics[0])

    # 6. save results as json
    #df.to_json("CMS-2022-0113_Unique-Letters_Predeistions.json", orient="records", lines=True)

if __name__ == "__main__":
    main()
