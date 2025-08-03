from fire import Fire
import pandas as pd
import openai, json
import pandas as pd, pickle
import os, sys
import dotenv

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

sys.path.append('..')
dotenv.load_dotenv()

client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def get_response( message ): 

	while True:
		response = client.chat.completions.create(
		model="chatgpt-4o-latest",
		# logprobs = True,
		messages=[
			{"role": "user", "content": message},
		],

		)
		try:
			x = json.loads(response.choices[0].message.content)
		except json.JSONDecodeError as e:
			continue

		return response.choices[0].message.content


def generate_adjectives(output_file: str = "adjectives.csv",
                        iterations: int = 4) -> None:

    prompts = ["\I am developing a Concept Bottleneck Model (CBM) for text classification and need a curated and extensive list of adjectives that can be used to describe the intentions expressed in a text. The task to be solved is counterspeech recognition. Provide only the list formatted as a JSON array in a single line.",
            "\I am developing a Concept Bottleneck Model (CBM) for text classification and need a curated and extensive list of adjectives that can be used to describe the intentions expressed in a text. The task to be solved is counterspeech recognition. Given this list of adjectives, I need you to extend it with additional adjectives that are relevant to the task.\n<adjective_list> {thelist} <adjectvie_list_end>\n\nProvide only the extended list formatted as a JSON array in a single line.",]
            
    response = get_response(prompts[0])

    lists = [json.loads(response)]
    
    print(f"Initial list of adjectives: {lists[-1]}")

    for i in range(iterations):
        lists += [json.loads(get_response(prompts[1].format(thelist=[j for i in lists for j in i])))]
        print(f"Iteration {i+1} list of adjectives:\n{lists[-1]}")

    lists += [[j for i in lists for j in i]]
    print(f"Retrieved: {len(lists[-1])} adjectives. {len(set(lists[-1]))} unique adjectives.")


    with open(output_file.replace('.csv', '-iterations.pkl'), "wb") as f:
        pickle.dump(lists, f)

    for i in range(len(lists)):
        #accumulate from the past
        if i > 0:
            lists[i] = lists[i-1] + lists[i]  # Accumulate adjectives from previous iterations
        lists[i] = list(set(lists[i]))  # Remove duplicates in each list
        
    df = {"adjective": lists[-1]}
    df = pd.DataFrame(df)
    df.to_csv(output_file, index=False) #save the adjectives to a csv file in a compatible format for our method to extract text representations
    return lists


def matching_vs_gold_standard(gold_standard_file: str = "adjectives.csv",
                               generated_per_iteration: list[list[str]] = None) -> None:

    """This function compares the generated adjectives with a gold standard file and calculates the matching ratio"""

    def get_matchings(to_cover_b: list[str], covering_b: list[str]) -> float:

        def extend_concepts(concepts):
            extended_concepts = []
            for concept in concepts:
                extended_concepts += [[concept]]
                for ss in wn.synsets(concept):
                    extended_concepts[-1] += ss.lemma_names()
            return extended_concepts

        def count_matches(covering, to_be_covered):

            lemmatizer = WordNetLemmatizer()

            lemmatized_covering = [lemmatizer.lemmatize(i) for i in covering]
            matches = 0
            for i in range(len(to_be_covered)):
                matches += lemmatizer.lemmatize(to_be_covered[i]) in lemmatized_covering
            
            return matches

        e_covering = extend_concepts(covering_b)
        e_covering = [j for i in e_covering for j in i]
        
        e_to_cover = extend_concepts(to_cover_b)
        assert len(e_to_cover) == len(to_cover_b)
        matches = count_matches(covering = covering_b, to_be_covered = to_cover_b)
        
        return matches/len(to_cover_b)

    df = pd.read_csv(gold_standard_file)
    our_adjectives = df['adjective'].tolist()

    print("How many of our adjectives are covered by gpt-4 adjectives")
    for i in range(len(generated_per_iteration)-1):
        e_m = get_matchings(to_cover_b=our_adjectives, covering_b=generated_per_iteration[i])
        print(f"Iteration {i} matchings: {e_m:.3f} ")


def main( output_file: str = "adjectives.csv",
          generation_iterations: int = 4,
          gold_standard_file: str = None) -> None:

     generated_per_iteration = generate_adjectives(output_file=output_file,
                                                   iterations=generation_iterations)
     matching_vs_gold_standard(gold_standard_file=gold_standard_file,
                               generated_per_iteration=generated_per_iteration)
     

if __name__ == "__main__":
     Fire(main)
