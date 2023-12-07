# This page contains the core logic of the Application

from myapp.models import AnalysisResult
from django.utils.timezone import make_aware
import pytz

def perform_analysis(file_path, classifiers):
    for r in range(2, len(classifiers) + 1):
        combinations_list = list(combinations(classifiers, r))

        for combo in combinations_list:
            num_classifiers = len(combo)
            results = train_and_evaluate_blended_classifier(combo, additional_info, num_classifiers)

            # Create an instance of the model and save it to the database
            result_instance = AnalysisResult(
                blended_classifiers=str(combo),
                accuracy=results['Accuracy'],
                kappa=results['Kappa'],
                precision=results['Precision'],
                recall=results['Recall'],
                f1_score=results['F1-Score'],
                confusion_matrix=str(results['Confusion Matrix']),
                execution_time=results['Execution Time (s)'],
                total_classifiers=results['Total Classifiers'],
                total_features=results['Total Features'],
                training_data_size=results['Training Data Size'],
                test_data_size=results['Test Data Size'],
                random_state=results['Random State'],
                preprocessing=results['Preprocessing'],
                smote=results['SMOTE'],
                total_cpu_cores=results['Total CPU Cores'],
                cpu_usage=results['CPU Usage (%)'],
                total_ram=results['Total RAM'],
                memory_usage=results['Memory Usage'],
                processor_type=results['Processor Type'],
                os_name=results['OS']
            )
            result_instance.save()
    return result

# Custom Tokenization function
def custom_tokenize(text):
    # Regular expressions to tokenize financial-specific terms and symbols
    # Example: Tokenize stock tickers like AAPL as single tokens
    text = re.sub(r'([A-Z]+)', r'\1 ', text)  # Add space after uppercase letters

    # Example: Tokenize currency symbols (e.g., $, €) and percentages
    text = re.sub(r'([$€%])', r' \1 ', text)  # Add space around currency symbols and percentages

    # Split the text into tokens using whitespace as the delimiter
    tokens = text.split()

    return tokens

# Data Preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters, URLs, and unnecessary symbols
    text = re.sub(r'http\S+', '', text).replace(r'www\S+', '')

    # Tokenization using custom tokenization function
    tokens = custom_tokenize(text)

    # Remove stopwords
    # custom_stopwords = set(custom_stopwords_df['Stopword'])
    # tokens = [word for word in tokens if word not in custom_stopwords]
    # stop_words = set(stopwords.words('english'))
    # tokens = [word for word in tokens if word not in stop_words]

    # Numerical handling (convert numbers to text representations)
    tokens = convert_numbers_to_text(tokens)

    # Join the words back to form preprocessed text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Function to convert numbers to text representations
def convert_numbers_to_text(tokens):
    converted_tokens = []
    for token in tokens:
        if token.replace('.', '', 1).isdigit():  # Check if it's a number
            converted_token = num2words(token)  # Convert number to text (e.g., "$1.50" to "one point five zero dollars")
            converted_tokens.append(converted_token)
        else:
            converted_tokens.append(token)
    return converted_tokens

# Function to convert bytes to MB or GB
def bytes_to_mb_or_gb(byte_value):
    if byte_value >= 1024**3:  # GB
        return f"{byte_value / (1024**3):.2f} GB"
    elif byte_value >= 1024**2:  # MB
        return f"{byte_value / (1024**2):.2f} MB"
    else:  # bytes
        return f"{byte_value} bytes"

def train_and_evaluate_blended_classifier(classifier_combo, additional_info, num_classifiers):
    print(f"Blending the following classifiers: {classifier_combo}")

    start_time = time.time()  # Record the start time
    start_time = time.time()
    # Get CPU information
    num_cores = multiprocessing.cpu_count()
    processor_type = platform.processor()
    # Get OS information
    os_name = platform.system()+'('+platform.release()+')'
    # Get RAM information
    ram_total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    ram_total_mb = ram_total / (1024**2)
    # Calculate CPU and RAM resources before training
    cpu_before = psutil.cpu_percent()
    available_memory_before = psutil.virtual_memory().available

################################################################################
    # Create the voting classifier with the selected combination
    selected_classifiers = [(name, classifiers[name]) for name in classifier_combo]
    voting_classifier = VotingClassifier(estimators=selected_classifiers, voting='hard')

    # Train the voting classifier
    voting_classifier.fit(X_train_tfidf_dense, y_train)

    # Make final predictions on the test data using the voting classifier
    final_predictions = voting_classifier.predict(X_test_tfidf_dense)

    # Calculate evaluation metrics for the blended model
    accuracy = accuracy_score(y_test, final_predictions)
    precision = precision_score(y_test, final_predictions, average='weighted', zero_division=1)
    recall = recall_score(y_test, final_predictions, average='weighted')
    f1 = f1_score(y_test, final_predictions, average='weighted')
    cm = confusion_matrix(y_test, final_predictions)
    kappa = cohen_kappa_score(y_test, final_predictions)
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    # Calculate CPU and RAM resources after training
    cpu_after = psutil.cpu_percent()
    available_memory_after = psutil.virtual_memory().available

    # Calculate resource usage during this iteration
    cpu_usage = max(cpu_after - cpu_before, 0)
    memory_usage = max(available_memory_after - available_memory_before, 0)  # Ensure non-negative value
    memory_usage_str = bytes_to_mb_or_gb(memory_usage)

    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"Memory Usage: {memory_usage_str}")

    # Store the results in the list as a dictionary
    results_dict = {
        'Blended Classifiers': classifier_combo,
        'Accuracy': accuracy,
        'Kappa': kappa,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm,
        'Execution Time (s)': execution_time,
        'Total Classifiers': additional_info['Total Classifiers'],
        'Total Features': additional_info['Total Features'],
        'Training Data Size': additional_info['Training Data Size'],
        'Test Data Size': additional_info['Test Data Size'],
        'Random State': additional_info['Random State'],
        'Preprocessing': additional_info['Preprocessing'],
        'SMOTE': additional_info['SMOTE'],
        'Total CPU Cores': num_cores,
        'CPU Usage (%)': cpu_usage,
        'Total RAM': ram_total_mb,
        'Memory Usage': memory_usage_str,
        'Processor Type': processor_type,
        'OS': os_name
    }
    print(results_dict)
    return results_dict

def Vectorizing_Balancing_Splitting():
    # Constants
    PREPROCESSING_FLAG = 1
    SMOTE_FLAG = 1
    # MAX_FEATURES_LIMIT = 15000
    TEST_SIZE = 0.2
    RAND_STATE = 42

    # Apply data preprocessing to the 'Sentence' column and create 'Preprocessed_Sentence' column
    df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text)

    # Prepare the data
    if PREPROCESSING_FLAG == 1:
      X = df['Preprocessed_Sentence']
    else:
      X = df['Sentence']

    y = df['Sentiment']

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer() # max_features=MAX_FEATURES_LIMIT
    X_train = tfidf_vectorizer.fit_transform(X)

    # Convert the labels to numerical values using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    smote = SMOTE(sampling_strategy='auto', random_state=RAND_STATE) # random_state=RAND_STATE
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_encoded)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_balanced, y_train_balanced, test_size=TEST_SIZE, random_state=RAND_STATE) # random_state=RAND_STATE

    # Convert X_train_tfidf to a dense numpy array
    X_train_tfidf_dense = X_train.toarray()

    # Convert X_test_tfidf to a dense numpy array
    X_test_tfidf_dense = X_test.toarray()

########################################################################################################################################
