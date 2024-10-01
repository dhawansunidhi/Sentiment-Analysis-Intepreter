import os
import re
import html
import torch
from spacy.lang.en import English
from collections import Counter
from torch.utils.data import Dataset

nlp = English()
PADDING_VALUE = 0
UNK_VALUE     = 1

class ReviewDataset(Dataset):
  """
  This class takes a Pandas DataFrame and wraps in a PyTorch Dataset.
  Read more about Torch Datasets here:
  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
  """

  def __init__(self, vocab, df, max_length=50):
    """
    Initialize this class with appropriate instance variables

    We would *strongly* recommend storing the dataframe itself as an instance
    variable, and keeping this method very simple. Leave processing to
    __getitem__.

    Sometimes, however, it does make sense to preprocess in __init__. If you
    are curious as to why, read the aside at the bottom of this cell.
    """

    self.df = df
    self.vocab = vocab
    self.max_length = max_length

  def __len__(self):
    """
    Return the length of the dataframe instance variable
    """

    df_len = len(self.df)

    return df_len

  def __getitem__(self, index: int):
    """
    Converts a dataframe row (row["tokenized"]) to an encoded torch LongTensor,
    using our vocab map created using generate_vocab_map. Restricts the encoded
    headline length to max_length.

    The purpose of this method is to convert the row - a list of words - into
    a corresponding list of numbers.

    i.e. using a map of {"hi": 2, "hello": 3, "UNK": 0}
    this list ["hi", "hello", "NOT_IN_DICT"] will turn into [2, 3, 0]

    Returns:
      tokenized_word_tensor (torch.LongTensor):
        A 1D tensor of type Long, that has each token in the dataframe mapped to
        a number. These numbers are retrieved from the vocab_map we created in
        generate_vocab_map.

        **IMPORTANT**: if we filtered out the word because it's infrequent (and
        it doesn't exist in the vocab) we need to replace it w/ the UNK token.

      curr_label (int):
        Binary 0/1 label retrieved from the DataFrame.

    """

    curr_row = self.df.iloc[index]
    curr_label = curr_row["label"]
    tokenized_word_list = curr_row["tokenized"]
    token_indices = [self.vocab.get(token, self.vocab["UNK"]) for token in tokenized_word_list]
    if len(token_indices) > self.max_length:
        token_indices = token_indices[:self.max_length]
    else:
        token_indices += [self.vocab[""]] * (self.max_length - len(token_indices))
    tokenized_word_tensor = torch.tensor(token_indices, dtype=torch.long)

    return tokenized_word_tensor, curr_label


def load_data(path, file_list, dataset, encoding='utf8'):
    """Read set of files from given directory and save returned lines to list.
    
    Parameters
    ----------
    path : str
        Absolute or relative path to given file (or set of files).
    file_list: list
        List of files names to read.
    dataset: list
        List that stores read lines.
    encoding: str, optional (default='utf8')
        File encoding.
        
    """
    for file in file_list:
        with open(os.path.join(path, file), 'r', encoding=encoding) as text:
            dataset.append(text.read())

#Cleaning Reviews
def spec_add_spaces(t: str) -> str:
    "Add spaces around / and # in `t`. \n"
    return re.sub(r"([/#\n])", r" \1 ", t)

def rm_useless_spaces(t: str) -> str:
    "Remove multiple spaces in `t`."
    return re.sub(" {2,}", " ", t)

def replace_multi_newline(t: str) -> str:
    return re.sub(r"(\n(\s)*){2,}", "\n", t)

def fix_html(x: str) -> str:
    "List of replacements from html strings in `x`."
    re1 = re.compile(r"  +")
    x = (
        x.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace(" @,@ ", ",")
        .replace("\\", " \\ ")
    )
    return re1.sub(" ", html.unescape(x))

def clean_text(input_text):
    text = fix_html(input_text)
    text = replace_multi_newline(text)
    text = spec_add_spaces(text)
    text = rm_useless_spaces(text)
    text = text.strip()
    return text

def tokenize(review):
    """
    This method take a text review and returns token 
    """
    return [token.text for token in nlp.tokenizer(review)]

def generate_vocab_map(df, cutoff=2):
    """
    This method takes a dataframe and builds a vocabulary to unique number map.
    It uses the cutoff argument to remove rare words occuring <= cutoff times.
    *NOTE*: "" and "UNK" are reserved tokens in our vocab that will be useful
    later. You'll also find the Counter imported for you to be useful as well.

    Args:
      df (pd.DataFrame): The entire dataset this mapping is built from
      cutoff (int): We exclude words from the vocab that appear less than or
                    eq to cutoff

    Returns:
      vocab (dict[str, int]):
        In vocab, each str is a unique token, and each dict[str] is a
        unique integer ID. Only elements that appear > cutoff times appear
        in vocab.

      reversed_vocab (dict[int, str]):
        A reversed version of vocab, which allows us to retrieve
        words given their unique integer ID. This map will
        allow us to "decode" integer sequences we'll encode using
        vocab!
    """

    vocab          = {"": PADDING_VALUE, "UNK": UNK_VALUE}
    reversed_vocab = None

    token_counts = Counter()
    for tokens in df["tokenized"]:
        token_counts.update(tokens)
    curr_index = 2
    for token, count in token_counts.items():
        if count > cutoff:
            vocab[token] = curr_index
            curr_index += 1
    reversed_vocab = {index: token for token, index in vocab.items()}

    return vocab, reversed_vocab

    