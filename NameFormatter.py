import os
import spacy
from dotenv import load_dotenv
from openai import OpenAI
from typing import List
from nameparser import HumanName
from cleantext import clean

class NameFormat:
    """
    A class that represents the text format.
    """

    def __init__(self, openAIKey, gpt_model="gpt-4-turbo-preview",SpacyModel="en_core_web_trf") -> None:
        """
        Initializes the class with the provided parameters.

        Args: open_ai_key (str): The API key for accessing the OpenAI service.
         gpt_model (str, optional): The name of the GPT
         spacy model to use. Defaults to "en_core_web_trf".
        model to use. Defaults to "gpt-4-1106-preview".

        Returns:
            None
            :type openAIKey: str

        """
        self.gpt_model = gpt_model
        self.open_ai_key = openAIKey
        self.client = OpenAI(api_key=self.open_ai_key)
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
    open_ai_key = os.getenv("OPENAI_API_KEY")

    textFormat = NameFormat(openAIKey=open_ai_key, )

    # Example usage
    input_data = ["Jennifer J. Messick, MRP ABR GRI AHWD", "Professor Mariann / Maz Hardey 💪🏼🦾🧠🎓🤳🏽🏃🏽‍♀️📘",
                  "Isaac Hammelburger, SEO", "Adam Bin Mahmood Baqashaim",
                  "Toby%20Juanes juanes", "Rob Hawes ACII, Chartered Insurer", "Adam “Ben” Clayton",
                  "Tony Garcia, ACNP, FNP, CNS, LP", "Dr. Nadja Wunderlich", "Noman Shahzad/ADM/SILMDK"]

    input_name = 'Rob Hawes ACII, Chartered Insurer'
    # clean_name = textFormat.cleanText(input_name)
    # print(f"The cleaned name is: {clean_name} \n")
    # print("Extracted names are: \n")
    # print(textFormat.extractNames(clean_name))
    # print ('\n')
    print('GPT-4 output\n:')
    prompt = """ can you write code for zero shot classification using facebook/bart-large-mnli using pytorch in optimised way for predicting in batches
"""


    print(textFormat.textGeneration(prompt))

