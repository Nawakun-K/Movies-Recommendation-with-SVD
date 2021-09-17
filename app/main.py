from loguru import logger

import pandas as pd
from services.load_data import LoadNetflixData

@logger.catch
def main(
    movies_title_path='./assets/movie_titles.csv'
    ):

    # Load NetFlix Data
    df = LoadNetflixData().load()

    # Create Movie Mapping df
    df_title = pd.read_csv(movies_title_path, encoding = "ISO-8859-1", header=None, names = ['Movie_Id', 'Year', 'Name'])
    df_title.set_index('Movie_Id', inplace=True)

if __name__ == "__main__":
    main()