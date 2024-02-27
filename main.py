import os
import spacy
import vertexai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
from nameparser import HumanName
from cleantext import clean
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.language_models import TextEmbeddingModel

class TextFormat:
    """
    A class that represents the text format.
    """

    def __init__(self, openAIKey, project_id, location, embedding_model="textembedding-gecko@003",gpt_model="gpt-4-1106-preview",SpacyModel="en_core_web_trf") -> None:
        """
        Initializes the class with the provided parameters.

        Args: open_ai_key (str): The API key for accessing the OpenAI service. project_id (str): The ID of the
        project. location (str): The location of the project. embedding_model (str, optional): The name of the
        embedding model to use. Defaults to "textembedding-gecko@003".
        gpt_model (str, optional): The name of the GPT
        model to use. Defaults to "gpt-4-1106-preview".

        Returns:
            None
            :type openAIKey: str
            :type location: str
            :type project_id: str
        """
        self.gpt_model = gpt_model
        self.project_id = project_id
        self.location = location
        self.open_ai_key = openAIKey
        self.client = OpenAI(api_key=self.open_ai_key)
        self.vertexai = vertexai.init(project=self.project_id, location=self.location)
        self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model)
        self.spacy_model = SpacyModel
        self.nlp = spacy.load(self.spacy_model)
    def cleanText(self, text: str) -> str:
        """
        Clean the text using the clean-text library.

        Args:
            text (str): The input text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        clean_text = clean(text,
                           fix_unicode=True,  # fix various unicode errors
                           to_ascii=True,  # transliterate to closest ASCII representation
                           lower=False,  # lowercase text
                           no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                           no_urls=True,  # replace all URLs with a special token
                           no_emails=True,  # replace all email addresses with a special token
                           no_phone_numbers=True,  # replace all phone numbers with a special token
                           no_numbers=True,  # replace all numbers with a special token
                           no_digits=True,  # replace all digits with a special token
                           no_currency_symbols=True,  # replace all currency symbols with a special token
                           no_punct=True,  # remove punctuations
                           replace_with_punct=" ",  # instead of removing punctuations you may replace them
                           replace_with_url=" ",
                           replace_with_email=" ",
                           replace_with_phone_number=" ",
                           replace_with_number=" ",
                           replace_with_digit=" ",
                           replace_with_currency_symbol=" ",
                           lang="en"  # set to 'en' for English special handling
                           )
        return clean_text

    def extractNames(self, text: str) -> List[dict]:
        """
        Extract names from the text using spacy.

        Args:
            text (str): The input text to be extracted.

        Returns:
            List[str]: The extracted names.
        """
        extracted_names = []
        doc = self.nlp(text)
        # Extract entities identified as PERSON
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

        if names:
            hName = HumanName(names[0]).as_dict()
            hName['Input_name'] = text
            extracted_names.append(hName)  # Take the first PERSON entity if available

        return extracted_names


    def text_embedding(self, text: str) -> List[float]:
        """
        Calculate the text embedding using a Large Language Model.
        Args:
            text (str): The input text to be embedded.
        Returns:
            List[float]: The embedding vector of the input text.
        """
        embeddings = self.embedding_model.get_embeddings([text])
        vector = pd.DataFrame(embeddings)['values'][0]
        return vector

    def textGeneration(self, prompt: str) -> List[str]:
        """
        Generate text using a GPT-4 model.

        Args:
            text (str): The input text to be generated.

        Returns:
            List[str]: The generated text.
            :param prompt:
        """
        stream = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")

if __name__ == "__main__":
    load_dotenv()
    project_id = os.getenv("PROJECT_ID")
    location = os.getenv("LOCATION")
    open_ai_key = os.getenv("OPENAI_API_KEY")

    textFormat = TextFormat(openAIKey=open_ai_key, project_id=project_id, location=location)

    # Example usage
    input_data = ["Jennifer J. Messick, MRP ABR GRI AHWD", "Professor Mariann / Maz Hardey 💪🏼🦾🧠🎓🤳🏽🏃🏽‍♀️📘",
                  "Isaac Hammelburger, SEO", "Adam Bin Mahmood Baqashaim",
                  "Toby%20Juanes juanes", "Rob Hawes ACII, Chartered Insurer", "Adam “Ben” Clayton",
                  "Tony Garcia, ACNP, FNP, CNS, LP", "Dr. Nadja Wunderlich", "Noman Shahzad/ADM/SILMDK"]

    # input_name = 'Rob Hawes ACII, Chartered Insurer'
    # clean_name = textFormat.cleanText(input_name)
    # print(f"The cleaned name is: {clean_name} \n") # clean_name = textFormat.extractNames(clean_name)
    # print("Extracted names are: \n")
    # print(textFormat.extractNames(clean_name))
    # print ('\n')
    print('GPT-4 output:')
    # prompt =  f'''Extract the human name from the text = '{input_name}' and only return first name middle name and last name in dict format'''
    prompt = ''' Write a a introduction of project which will have following steps 
step 1 : cleaning Step , which cleans docs called as cleaning step
step 2 : this step predict the Language of doc and extract NER enitity called as Ner lang predictor
step 3 : This step consolidate the out of step 1 ,step 2 and explode the NER tags 
step 4 : This step is the begnining of stage 2. where we use resourefile to calculate the confidence of a product found in text
step 5 : This step is the final step which consolidates all the score at Domain level and calculate the HIGH , MEDIUM LOW  confidence 
answer in paragraph format'''
    print(textFormat.textGeneration(prompt))

