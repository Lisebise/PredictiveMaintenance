import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(filepath):
    """Loads the data from the csv file and stores it into a dataframe
    :param filepath: (string) containing the filepath of csv file
    :return df: (pd.DataFrame) dataframe with the loaded csv file
    """
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """
    :param df: (pd.DataFrame) dataframe that needs to be cleaned
    :return df: (pd.DataFrame) cleaned dataframe
    """
    assert len(np.unique(df["Product ID"].values))==df.shape[0], "Product ID could be relevant."
    # Product ID do not have anything to do with failure as they are as many uniques ones as rows -> drop it
    df = df.drop(columns=["Product ID"])
    # no catgorization possible with UDI -> drop it
    df = df.drop(columns=["UDI"])
    # add rotational speed * torque to dataframe to get power required for process
    df["Power [W]"] = df["Torque [Nm]"]*(df["Rotational speed [rpm]"]/0.10471976)# 1 r/min = 2π rad·min−1 = 2π/60 rad·s−1 ≈ 0.10471976 rad·s−1
    # only predict if it is power failure or other failure
    # therefor eliminate other failures and replace them
    categories = np.unique(df["Failure Type"].values)
    for failure in categories:
        if failure != "No Failure" and failure != "Power Failure":
            df = df.replace(failure, "Other Failure")
    return df


def save_data(df, database_filename):
    """Save the data to a sql database
    :param df: (pd.DataFrame) dataframe that will be saved in the given database
    :param database_filename: (string) name of the database with correct ending (.db)
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("PredicitiveMaintentace", con=engine, if_exists="replace", index=False)


def main():
    # load data
    df = load_data("predictive_maintenance.csv")
    # clean data
    df = clean_data(df)
    # save the data to a database
    save_data(df, "PredicitiveMaintentace")
    print("Cleaned data saved to database!")


if __name__ == '__main__':
    main()
