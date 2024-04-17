import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint

WORD_2_VEC_VECTOR_SIZE = 100

# Loading the dataset
print("Loading the dataset...")
df = pd.read_csv('patient_ehr.csv')
print("Dataset loaded.")




# Data preprocessing
print("Performing data preprocessing...")
# Convert date columns to datetime objects
df['last_visit_date'] = pd.to_datetime(df['last_visit_date'])


# Drop irrelevant columns and select relevant columns
selected_columns = ['age', 'blood_glucose_levels', 'cholesterol_levels', 'heart_rate']
categorical_columns = ['gender', 'education level', 'smoking_status',
					   'physical activity level', 'dietary habits', 'alcohol consumption',
					   'health status', 'family_history', 'air_pollution_levels',
					   'geographic_location', 'climate_conditions', 'mental_health_status',
					   'stress_levels', 'coping_mechanisms', 'adherence_to_treatment',
					   'healthcare_utilization_patterns', 'medication']

df = df.drop(columns=['patient_id', 'name'])
df.fillna('Unknown', inplace=True)



# Split the dataset into train and test sets
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
print("Dataset split successfully.")

# Convert categorical columns to lists of lists
categorical_data_train = X_train[categorical_columns].values.tolist()
categorical_data_test = X_test[categorical_columns].values.tolist()

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=categorical_data_train, vector_size=WORD_2_VEC_VECTOR_SIZE, window=5, min_count=1, workers=4)
print("Word2Vec model trained successfully.")


# Embed categorical variables

def word2vec_wv(w2v_model=None, category=None):
	try:
		return w2v_model.wv[category]
	except:
		return [0] * WORD_2_VEC_VECTOR_SIZE

# Create an empty dictionary to store rows for each chronic disease
chronic_disease_dict = {}

# Iterate over each chronic disease category
for chronic_disease in df['chronic_disease'].unique():
	# print("chronic_disease: ", chronic_disease)
	# Filter the DataFrame for rows with the current chronic disease
	chronic_disease_rows = df[df['chronic_disease'] == chronic_disease]
	# Drop the 'chronic_disease' column
	chronic_disease_rows = chronic_disease_rows.drop(columns=['chronic_disease'])
	# Store the filtered rows in the dictionary
	chronic_disease_dict[chronic_disease] = {}
	chronic_disease_dict[chronic_disease]['data_frame'] = chronic_disease_rows
	chronic_disease_dict[chronic_disease]['data_frame_categorical'] = chronic_disease_rows[categorical_columns].values.tolist()
	chronic_disease_dict[chronic_disease]['data_frame_selected'] = chronic_disease_rows[selected_columns].values.tolist()
	chronic_disease_dict[chronic_disease]['embedded_categorical_data'] = [[word2vec_wv(word2vec_model, category) for category in sample] for sample in chronic_disease_dict[chronic_disease]['data_frame_categorical']]
	chronic_disease_dict[chronic_disease]['combined_vectors'] = [chronic_disease_dict[chronic_disease]['data_frame_selected'][i] + [val for sublist in chronic_disease_dict[chronic_disease]['embedded_categorical_data'][i] for val in sublist] for i in range(len(chronic_disease_dict[chronic_disease]['data_frame_selected']))]
	# print(chronic_disease_dict[chronic_disease]['combined_vectors'][:1])
	# print("\n\n\n")

#################################




# [[embed1], [embed2], [embed3], [embed4], ...]

## CODE HERE
def predict_chronic_diseases(**kwargs):
	input_data = [kwargs[key] for key in selected_columns + categorical_columns]
	embedded_input = [[word2vec_wv(word2vec_model, category) for category in input_data[len(selected_columns):]]]
	new_input = input_data[:len(selected_columns)] + [val for sublist in embedded_input[0] for val in sublist]

	# chronic_disease_similarity = {}
	# iterate the chronic_disease_dict
		# chronic_disease_name
		# use 'chronic_disease_dict' and the 'combined_vectors' in them
		# chronic_disease_similarity[chronic_disease_name] = []
		# iterate over all the combined vectors
			# check the cosine similarity b/w new_input and the combined vector
			# append the similarity score in chronic_disease_similarity[chronic_disease_name] list
		# sort in decending order the chronic_disease_similarity[chronic_disease_name] list    
	# check which array in chronic_disease_similarity dict is having first top 10 items highest value when summed up.

	chronic_disease_similarity = {}

	# Iterate over each chronic disease category
	for chronic_disease, data in chronic_disease_dict.items():
		combined_vectors = data['combined_vectors']
		similarities = []

		# Calculate cosine similarity between new input and each combined vector
		for vector in combined_vectors:
			similarity_score = cosine_similarity([new_input], [vector])[0][0]
			similarities.append(similarity_score)

		# Sort similarities in descending order
		sorted_similarities = sorted(similarities, reverse=True)[:10]

		chronic_disease_similarity[chronic_disease] = sorted_similarities[0]
  

	# Identify the chronic disease with the highest sum of similarities
	predicted_chronic_disease = max(chronic_disease_similarity, key=chronic_disease_similarity.get)
	
	return predicted_chronic_disease

#################################



TEST_INDEX = 5 # Select the index of the test you want to run:

# Take a sample input from X_test dataset
sample_input = X_test.iloc[TEST_INDEX]

# Print the sample input
print("\n\nSample Input:")
pprint(sample_input.to_dict())

if input("\nDo you want to run a manual test on an auto taken sample input ? [y/n]").lower().strip().startswith('y'):
	# Retrieve the actual chronic disease label from the test dataset for the sample input
	actual_chronic_disease = X_test.iloc[TEST_INDEX]['chronic_disease']

	# Call the predict_chronic_diseases function on the sample input
	predicted_chronic_disease = predict_chronic_diseases(**sample_input.to_dict())

	# Print the predicted and actual chronic diseases
	print("\nPredicted Chronic Disease:", predicted_chronic_disease)
	print("Actual Chronic Disease:", actual_chronic_disease)

	# Verify if the prediction is correct
	if predicted_chronic_disease == actual_chronic_disease:
		print("Prediction is Correct!")
	else:
		print("Prediction is Incorrect!")

elif input("Do you want to input the data manually by hand for a test input ? [y/n]").lower().strip().startswith('y'):
    # Manually input the data for a test input
    sample_input = {}
    print("\nEnter the details for the test input:")
    for column in selected_columns + categorical_columns:
        value = input(f"Enter value for '{column}': ")
        sample_input[column] = value

    # Call the predict_chronic_diseases function on the sample input
    predicted_chronic_disease = predict_chronic_diseases(**sample_input)

    # Print the predicted chronic disease
    print("\nPredicted Chronic Disease:", predicted_chronic_disease)


if input("\nDo you want to check accuracy of model on test data ? [y/n] (Would need to wait for a bit) : ").strip().lower().startswith('y'):
	# Calculate accuracy on X_test data
	correct_predictions = 0
	total_samples = len(X_test)

	for index, row in X_test.iterrows():
		predicted_chronic_disease = predict_chronic_diseases(**row.to_dict())
		actual_chronic_disease = row['chronic_disease']
		if predicted_chronic_disease == actual_chronic_disease:
			print(f"predicted_chronic_disease {predicted_chronic_disease} == actual_chronic_disease {actual_chronic_disease}: ")
			correct_predictions += 1

	accuracy = (correct_predictions / total_samples) * 100
	print("Accuracy on X_test data:", accuracy, "%")

