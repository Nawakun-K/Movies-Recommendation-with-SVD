from loguru import logger

import pandas as pd
from services.load_data import LoadNetflixData
from services.svd_model import SVDModel

@logger.catch
def main(
    movies_title_path='./assets/movie_titles.csv',
    user_id=785314
    ):

    # Load NetFlix Data
    loader = LoadNetflixData()
    df = loader.load()

    # Create Movie Mapping df
    df_title = pd.read_csv(movies_title_path, encoding = "ISO-8859-1", header=None, names = ['Movie_Id', 'Year', 'Name'])
    df_title.set_index('Movie_Id', inplace=True)

    # Instantiate model
    svd = SVDModel()
    svd.fit(df)
    
    # Make prediction
    prediction = svd.predict(id=user_id)
    prediction = prediction.sort_values(by=['Estimate_Score'], ascending=False)
    prediction = prediction.merge(df_title, left_on='Movie_Id', right_index=True)
    prediction = prediction[['Movie_Id', 'Year', 'Name', 'Estimate_Score']]

    print(f"==========Movies that {user_id} is most likely to watch==========")
    print(prediction.head(10))

if __name__ == "__main__":
    main()