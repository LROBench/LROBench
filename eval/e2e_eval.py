import pandas as pd
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)
sys.path.append(project_root)

from src.operators.logical import (
    LogicalSelect,
    LogicalMatch,
    LogicalImpute,
    LogicalCluster,
    LogicalOrder,
    LogicalInduce,
)
from src.core.enums import OperandType, ImplType

# shipping
# select
def pipeline_0():
    query = "Which customers have shipments in June 2016, where the customer’s state is geographically adjacent to the shipment’s state?"
    answer = "[Eppies Discount Tire & Auto Centers, Kenner Welding Inc, Sunguard Window Tinting & Truck Accessories, R V City, Great Dane Trailers Inc]"

    shipment = pd.read_csv("./databases/shipping/shipment.csv")
    shipment = shipment[
        (pd.to_datetime(shipment["ship_date"]).dt.month == 6)
        & (pd.to_datetime(shipment["ship_date"]).dt.year == 2016)
    ]
    city = pd.read_csv("./databases/shipping/city.csv")
    customer = pd.read_csv("./databases/shipping/customer.csv")
    shipment = pd.merge(shipment, city, on="city_id")
    shipment = pd.merge(shipment, customer, on="cust_id")

    df = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The two states are geographically adjacent.",
        df=shipment,
        depend_on=["state_x", "state_y"],
        thinking=True,
    )
    print(df[["state_x", "state_y"]])
    pred = df["cust_name"].drop_duplicates()
    return pred


# shipping
# select + select
def pipeline_1():
    query = "How many shipments did northeast USA customers make before the opening ceremony of the 2016 Olympics?"
    answer = 12
    shipment = pd.read_csv("./databases/shipping/shipment.csv")
    customer = pd.read_csv("./databases/shipping/customer.csv")
    customer = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The customer is located in the northeast side of USA",
        df=customer,
        depend_on=["city", "state"],
        thinking=True,
    )
    print(customer)
    df = pd.merge(shipment, customer, on="cust_id")
    print(df)
    df = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The shipment happened before the opening ceremony of the 2016 Olympics",
        df=df,
        depend_on=["ship_date"],
        thinking=True,
    )
    return df.shape[0]


# restaurants2
# select + impute
def pipeline_2():
    query = "Infer the region of origin of cuisines served at popular Yelp restaurants (votes > 3000) in the Northeastern city of the United States?"
    answer = [['China'], ['Global'], ['Italy'], ['Japan'], ['Italy'], ['United States'], ['Italy'], ['United States'], ['Japan']]
    yelp = pd.read_csv("./databases/restaurants2/yelp.csv")
    yelp = yelp[yelp["votes"] > 3000]
    print(len(yelp))
    yelp = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The zip belongs to a northeast city of USA",
        df=yelp,
        depend_on=["zip"],
        thinking=True,
    )
    print(len(yelp))
    print(yelp)

    result = LogicalImpute(operand_type=OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Impute the cuisine's region of origin.",
        df=yelp,
        depend_on="cuisine",
        new_col="ori_region",
    )
    print(result)
    prediction = result[['ori_region']].values.tolist()
    return prediction


# restaurants2
# select
def pipeline_3():
    query = "How many restaurants in zomato with a rating no less than 4.7 serve seafoods?"
    answer = 10
    zomato = pd.read_csv("./databases/restaurants2/zomato.csv")

    zomato = zomato[zomato["rating"] >= 4.7]
    print(zomato)
    print(len(zomato))
    zomato = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The restaurant serves seafood.",
        df=zomato,
        depend_on=["cuisine"],
        thinking=True,
    )
    print(zomato['cuisine'])
    return zomato.shape[0]


# superhero
# select
def pipeline_4():
    query = "List DC Comics superheroes that are taller than Michael Jordan, have red eyes, and possess a power related to agility?"
    answer = "['Amazo', 'Ares', 'Darkseid', 'Doomsday', 'Killer Croc', 'Martian Manhunter']"

    hero = pd.read_csv("./databases/superhero/superhero.csv")
    publisher = pd.read_csv("./databases/superhero/publisher.csv")
    power = pd.read_csv("./databases/superhero/superpower.csv")
    hero_power = pd.read_csv("./databases/superhero/hero_power.csv")
    colour = pd.read_csv("./databases/superhero/colour.csv")

    publisher = publisher[publisher["publisher_name"] == "DC Comics"]
    colour = colour[colour["colour"] == "Red"]

    merged_df = pd.merge(
        hero.rename(columns={"id": "hero_id"}),
        publisher.rename(columns={"id": "publisher_id"}),
        on="publisher_id",
    )
    merged_df = pd.merge(merged_df, hero_power, on="hero_id")
    merged_df = pd.merge(
        merged_df, power.rename(columns={"id": "power_id"}), on="power_id"
    )
    merged_df = pd.merge(
        merged_df,
        colour.rename(columns={"id": "eye_colour_id", "colour": "eye_colour"}),
        on="eye_colour_id",
    )

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This power is related to agility.",
        df=merged_df,
        depend_on=["power_name"],
        thinking = True,
    )

    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This hero is higher than Michael Jordan.",
        df=result,
        depend_on=["height_cm"],
        thinking=True,
    )
    result = result[["superhero_name"]].drop_duplicates()

    return result.tolist()


# formula_1
# groupby
def pipeline_5():
    query = "Classify circuits which host Grand Prix in 2016 by their location continent. Answer with the location and corresponding continent."
    answer = "[['Kuala Lumpur', 'Asia'], ['Sakhir', 'Asia'], ['Montmeló', 'Europe'], ['Monte-Carlo', 'Europe'], ['Montreal', 'North America'], ['Silverstone', 'Europe'], ['Hockenheim', 'Europe'], ['Budapest', 'Europe'], ['Spa', 'Europe'], ['Monza', 'Europe'], ['Marina Bay', 'Asia'], ['Shanghai', 'Asia'], ['São Paulo', 'South America'], ['Suzuka', 'Asia'], ['Abu Dhabi', 'Asia'], ['Mexico City', 'North America'], ['Austin', 'North America'], ['Spielburg', 'Europe'], ['Sochi', 'Europe'], ['Baku', 'Asia']]"

    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    merged_df = pd.merge(
        circuits.rename(
            columns={"name": "circuit_name", "country": "location_country"}
        ),
        races,
        on="circuitId",
    )
    merged_df = merged_df[merged_df["year"] == 2016]

    op = LogicalCluster(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Cluster the data by the continent in which the circuit locates in",
        df=merged_df,
        depend_on="location_country", 
        thinking = True,
    )
    print(result.columns)
    result = result[["location", "cluster_name"]].values.tolist()

    return result


# formula_1
# induce
def pipeline_6():
    query = "Summarize performance of drivers who have Top10 fastest average lap time in 2016 Malaysia Grand Prix."
    answer = """
        The top 10 drivers by average lap time at the 2016 Malaysia Grand Prix are listed below, with each entry summarizing the race result and average race pace:

        - Daniel Ricciardo (AUS) — Winner; avg lap 1:44.157. Managed tyre strategy and held off Verstappen after Hamilton’s retirement.
        - Max Verstappen (NED) — P2; avg lap 1:44.200. Very close pace to Ricciardo, completing a Red Bull 1–2.
        - Nico Rosberg (GER) — P3; avg lap 1:44.434. Recovered from a lap‑1 spin after contact and a 10s penalty to reach the podium.
        - Kimi Räikkönen (FIN) — P4; avg lap 1:44.671. Solid run, just behind Rosberg on pace.
        - Valtteri Bottas (FIN) — P5; avg lap 1:45.256. Best of the rest.
        - Sergio Pérez (MEX) — P6; avg lap 1:45.296. Consistent, closely matched to Bottas on pace.
        - Fernando Alonso (ESP) — P7; avg lap 1:45.321. Strong climb through the field after starting at the back due to penalties.
        - Nico Hülkenberg (GER) — P8; avg lap 1:45.479. Competitive midfield run.
        - Jenson Button (GBR) — P9; avg lap 1:45.618. Scored points in his 300th GP.
        - Lewis Hamilton (GBR) — DNF (engine failure while leading); fastest avg lap 1:43.314, indicating top pace before retirement.
        
        Takeaways: Hamilton had the strongest average pace but retired; Red Bull’s duo turned steady 1:44‑low pace into a 1–2; Rosberg’s recovery and Ferrari’s Kimi followed; and the midfield (Williams/Force India/McLaren) clustered in the mid‑1:45s.
    """

    races = pd.read_csv("./databases/formula_1/races.csv")
    lapTimes = pd.read_csv("./databases/formula_1/lapTimes.csv")
    drivers = pd.read_csv("./databases/formula_1/drivers.csv")

    races = races[(races["year"] == 2016) & (races["name"] == "Malaysian Grand Prix")]
    merged_df = pd.merge(races, lapTimes, on="raceId")
    merged_df = (
        merged_df.groupby("driverId")
        .agg(avgLapTime=("milliseconds", "mean"))
        .reset_index()
    )

    merged_df = merged_df.sort_values(by="avgLapTime", ascending=True)[:10]
    merged_df = pd.merge(merged_df, drivers, on="driverId")
    merged_df = merged_df[["forename", "surname", "nationality", "avgLapTime"]]
    print(merged_df)

    op = LogicalInduce(operand_type=OperandType.ROW)
    result = op.execute(
        ImplType.LLM_ALL,
        condition="Summarize the performance of those drivers's performance in 2016 Malaysia Grand Prix.",
        df=merged_df,
        thinking=True,
    )

    return result


# california_schools
# select
def pipeline_7():
    query = "How many charter schools are located north than Sacramento have been closed?"
    answer = 32

    schools = pd.read_csv("./databases/california_schools/schools.csv")
    schools = schools[(schools["StatusType"] == "Closed") & (schools["Charter"] == 1)]

    schools = schools[["CDSCode", "Latitude"]]
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This schools is located north than Sacramento, CA.",
        df=schools,
        depend_on=["Latitude"],
    )
    # result = result[['CDSCode']].drop_duplicates()

    return len(result)


# california_schools
# select
def pipeline_8():
    query = "List School that posesses Top 30 FRPM counts with a human-sounding name."
    answer = ["John H. Francis Polytechnic", "Hector G. Godinez", "James A. Garfield Senior High"]

    schools = pd.read_csv("./databases/california_schools/schools.csv")
    frpm = pd.read_csv("./databases/california_schools/frpm.csv")

    merged_df = pd.merge(schools, frpm, on="CDSCode")
    merged_df = merged_df.dropna(subset=["FRPM Count (K-12)"])
    merged_df = merged_df.sort_values(by="FRPM Count (K-12)", ascending=False)[:30]
    merged_df = merged_df[["County", "School Name"]]

    op = LogicalSelect(operand_type=OperandType.ROW)

    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This schools sounds like a human's name.",
        df=merged_df,
        depend_on=["School Name"],
    )
    result = result[["School Name"]]
    return result


# california_schools
# select + order
def pipeline_9():
    query = "List all counties that have an Elementary school with an Percent (%) Eligible FRPM (K-12) less than 1%, and sort them by population in descending order."
    answer = ["Los Angeles", "Contra Costa", "Marin", "Santa Clara", "San Mateo", "Alameda"]

    schools = pd.read_csv("./databases/california_schools/schools.csv")
    frpm = pd.read_csv("./databases/california_schools/frpm.csv")

    merged_df = pd.merge(schools, frpm, on="CDSCode")
    merged_df = merged_df.dropna(subset=["FRPM Count (K-12)"])
    merged_df = merged_df[merged_df["Percent (%) Eligible FRPM (K-12)"] < 0.01]

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This school is an Elementary school.",
        df=merged_df,
        depend_on=["School Name"],
    )

    result = result[["County"]].drop_duplicates()
    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(result)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="the population of county listed in the 'County' column",
        depend_on="County",
        ascending=False,
        k=k,
        df=result,
        sort_algo="heap",
    )
    return result


# electronics
# select + select
def pipeline_10():
    query = "Among Top 10 expensive electronic products sold on Amazon, how many of their manufacture originates outside USA and has a 8GB memory."
    answer = 4

    amazon_product = pd.read_csv("./databases/electronics/amazon.csv")

    amazon_product = amazon_product.dropna(subset=["Amazon_Price"])
    amazon_product = amazon_product.sort_values(by="Amazon_Price", ascending=False)[:10]

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This product's manufacture originates outside USA.",
        df=amazon_product,
        depend_on=["Brand"],
    )

    result = result[["ID", "Features"]]

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This product's has 8GB memory.",
        df=result,
        depend_on=["Features"],
    )

    result = result[["ID"]]
    return len(result)


# debit_card_specializing
# select
def pipeline_11():
    query = "How many transactions in 2012-08-26 happened in CZE payed in CZK are larger than 10 US dollars."
    answer = 49

    transactions = pd.read_csv(
        "./databases/debit_card_specializing/transactions_1k.csv"
    )
    gasstations = pd.read_csv("./databases/debit_card_specializing/gasstations.csv")
    customers = pd.read_csv("./databases/debit_card_specializing/customers.csv")
    customers = customers[customers["Currency"] == "CZK"]
    gasstations = gasstations[gasstations["Country"] == "CZE"]

    merged_df = pd.merge(transactions, customers, on="CustomerID")
    merged_df = pd.merge(merged_df, gasstations, on="GasStationID")
    merged_df["Date"] = pd.to_datetime(
        merged_df["Date"], errors="coerce", format="%Y-%m-%d"
    )
    merged_df = merged_df[merged_df["Date"] == "2012-08-26"]
    # merged_df = merged_df[merged_df['Price'] > 10*20.99]

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The price in CZK as currency is larger than 10 USD.",
        df=merged_df,
        depend_on=["Price"],
        thinking = True,
    )
    result = result[["TransactionID"]].drop_duplicates()

    return len(result)


# debit_card_specializing
# impute
def pipeline_12():
    query = "What is the average gas consumption in Leap Year and Common Year respectively."
    answer = [13403, 2562]

    consumption = pd.read_csv("./databases/debit_card_specializing/yearmonth.csv")
    date_copy = consumption.copy()
    date_copy = date_copy[["Date"]].drop_duplicates()

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Impute the year attribute of the input date. Options are 'Leap Year' or 'Common Year'.",
        df=date_copy,
        depend_on="Date",
        new_col="YearAttr",
    )
    merged_df = pd.merge(consumption, result, on="Date")

    result = (
        merged_df.groupby("YearAttr")
        .agg(avgConsumption=("Consumption", "mean"))
        .reset_index()
    )

    return result["avgConsumption"].tolist()


# car_retails
# select
def pipeline_13():
    query = "How many customers living in North America have spent half of their credit card limit for car payments?"
    answer = 18
    customers = pd.read_csv("./databases/car_retails/customers.csv")
    payments = pd.read_csv("./databases/car_retails/payments.csv")

    merged_df = pd.merge(customers, payments, on="customerNumber")
    merged_df = merged_df[merged_df["amount"] > 0.5 * merged_df["creditLimit"]]
    merged_df = merged_df[["amount", "creditLimit", "country", "customerNumber"]]

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This country locates in North America.",
        df=merged_df,
        depend_on=["country"],
    )

    result = result[["customerNumber"]].drop_duplicates()

    return len(result)


# car_retails
# cluster
def pipeline_14():
    query = "Count how many Vintage Cars for sales are produced from 'USA', 'Europe' and 'Others' respectively."
    answer = [18, 5, 1]
    products = pd.read_csv("./databases/car_retails/products.csv")
    productLines = pd.read_csv("./databases/car_retails/productlines.csv")

    productLines = productLines[productLines["productLine"] == "Vintage Cars"]
    merged_df = pd.merge(products, productLines, on="productLine")
    merged_df = merged_df[["productName"]]

    op = LogicalCluster(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Cluster the product data by their manufacture's country into 'USA', 'Europe', 'Others'.",
        df=merged_df,
        depend_on="productName",
    )

    result = (
        result.groupby("cluster_name").agg(count=("productName", "count")).reset_index()
    )

    return result


# car_retails
# select
def pipeline_15():
    query = "How many customer commented on shipped orders to request specific company for delivering."
    answer = 12
    orders = pd.read_csv("./databases/car_retails/orders.csv")
    customer = pd.read_csv("./databases/car_retails/customers.csv")
    orders = orders[orders["status"] == "Shipped"]
    orders = orders.dropna(subset=["comments"])
    merged_df = pd.merge(customer, orders, on='customerNumber')

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This comment appoints or requests specific company for delivering.",
        df=merged_df,
        depend_on=["comments"],
        thinking = True,
    )
    result = result[["customerNumber"]].drop_duplicates()

    return len(result)


# chicago_crime
# select
def pipeline_16():
    query = "Among crimes happened in the Top 10 northernest place in Chicago, how many of them are reported at night(19 pm to 6 am)?"
    answer = 3

    crime = pd.read_csv("./databases/chicago_crime/Crime.csv")
    crime = crime.sort_values(by="latitude", ascending=False)[:10]

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This crime happens at night (19 pm to 6 am).",
        df=crime,
        depend_on=["date"],
    )
    result = result[["date", "latitude", "report_no"]].drop_duplicates()
    return len(result)


# chicago_crime
# select
def pipeline_17():
    query = "Among Chicago wards that is responsible for crimes in 'Central' district, how many of them are democrates?"
    answer = 8
    crime = pd.read_csv("./databases/chicago_crime/Crime.csv")
    ward = pd.read_csv("./databases/chicago_crime/Ward.csv")
    district = pd.read_csv("./databases/chicago_crime/District.csv")

    district = district[district["district_name"] == "Central"]
    merged_df = pd.merge(crime, district, on="district_no")

    merged_df = pd.merge(merged_df, ward, on="ward_no")
    merged_df = merged_df[
        ["alderman_first_name", "alderman_last_name"]
    ].drop_duplicates()

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This Chicago ward is a Democrat.",
        df=merged_df,
        depend_on=["alderman_first_name", "alderman_last_name"],
    )

    return len(result)


# chicago_crime
# select
def pipeline_18():
    query = "How many FBI agents are responsible for investigating sexual crimes?"
    answer = 3

    fbi = pd.read_csv("./databases/chicago_crime/FBI_Code.csv")
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This FBI agent is responsible for investigating sexual crimes.",
        df=fbi,
        depend_on=["description"],
    )

    return len(result)


# codebase_community
# select
def pipeline_19():
    query = "In the Top 10 popular posts, List post id of those have tags related to proramming language."
    answer = [12398]

    posts = pd.read_csv("./databases/codebase_community/posts.csv")
    posts = posts.sort_values(by="ViewCount", ascending=False)[:10]
    posts = posts[["Id", "Tags"]]

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="These tags are related to proramming language.",
        df=posts,
        depend_on=["Tags"],
        thinking = True,
    )
    result = result["Id"].tolist()

    return result


# codebase_community
# select
def pipeline_20():
    query = "How many AboutMe description is not null and contains external link among Top 200 newest registered users."
    answer = 7

    users = pd.read_csv("./databases/codebase_community/users.csv")
    users = users.sort_values(by="CreationDate", ascending=False)[:200]
    users = users.dropna(subset=["AboutMe"])
    users = users[["Id", "AboutMe"]]

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This personal introudction contains external link.",
        df=users,
        depend_on=["AboutMe"],
    )
    result = result[["Id", "AboutMe"]]

    return len(result)


# books
# select + select
def pipeline_21():
    query = "How many books are written in a language originates from an Asian country, and are bought by customers using a government affiliated email?"
    answer = 2
    customer = pd.read_csv("./databases/books/customer.csv")
    cust_order = pd.read_csv("./databases/books/cust_order.csv")
    book = pd.read_csv("./databases/books/book.csv")
    book_language = pd.read_csv("./databases/books/book_language.csv")
    order_line = pd.read_csv("./databases/books/order_line.csv")

    op = LogicalSelect(operand_type=OperandType.ROW)
    book_language = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This language originates from an Asian country.",
        df=book_language,
        depend_on=["language_name"],
        thinking = True,
    )

    customer = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This is a government affiliated email.",
        df=customer,
        depend_on=["email"],
        thinking = True,
    )

    merged_df = pd.merge(customer, cust_order, on="customer_id")
    merged_df = pd.merge(merged_df, order_line, on="order_id")
    merged_df = pd.merge(merged_df, book, on="book_id")
    merged_df = pd.merge(merged_df, book_language, on="language_id")
    merged_df = merged_df[["book_id"]].drop_duplicates()

    prediction = len(merged_df)
    return prediction


# books
# select
def pipeline_22():
    query = "How many book are written in English or its variation?"
    answer = 10544
    book = pd.read_csv("./databases/books/book.csv")
    book_language = pd.read_csv("./databases/books/book_language.csv")

    op = LogicalSelect(operand_type=OperandType.ROW)
    book_language = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This book is written in English or its variation.",
        df=book_language,
        depend_on=["language_name"],
    )
    merged_df = pd.merge(book, book_language, on="language_id")
    merged_df = merged_df[["book_id"]].drop_duplicates()
    prediction = len(merged_df)

    return prediction


# books
# select
def pipeline_23():
    query = "How many books written by Mark Twain are sold to customers living in North America."
    answer = 2
    country = pd.read_csv("./databases/books/country.csv")
    customer = pd.read_csv("./databases/books/customer.csv")
    address = pd.read_csv("./databases/books/address.csv")
    customer_address = pd.read_csv("./databases/books/customer_address.csv")
    cust_order = pd.read_csv("./databases/books/cust_order.csv")
    order_line = pd.read_csv("./databases/books/order_line.csv")
    book = pd.read_csv("./databases/books/book.csv")
    author = pd.read_csv("./databases/books/author.csv")
    book_author = pd.read_csv("./databases/books/book_author.csv")

    op = LogicalSelect(operand_type=OperandType.ROW)
    country = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This country locates in North America.",
        df=country,
        depend_on=["country_name"],
        thinking = True,
    )

    author = author[author["author_name"] == "Mark Twain"]
    book_full = pd.merge(book, book_author, on="book_id")
    book_full = pd.merge(book_full, author, on="author_id")

    customer_full = pd.merge(customer, customer_address, on="customer_id")
    customer_full = pd.merge(customer_full, address, on="address_id")
    customer_full = pd.merge(customer_full, country, on="country_id")

    merged_df = pd.merge(customer_full, cust_order, on="customer_id")
    merged_df = pd.merge(merged_df, order_line, on="order_id")
    merged_df = pd.merge(merged_df, book_full, on="book_id")

    merged_df = merged_df[["book_id"]].drop_duplicates()

    prediction = len(merged_df)

    return prediction


# movies
# order
def pipeline_24():
    query = "Rank Top 5 highest rated movies in imdb by their box office from highest to lowest, and return their ID accordingly."
    answer = ["a-1493", "a-701", "a-362", "a-2007", "a-2148"]
    imdb = pd.read_csv("./databases/movies/imdb.csv")
    imdb = imdb.sort_values(by="Rating", ascending=False)[:5]

    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(imdb)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="box office of this movie",
        depend_on="Title",
        ascending=False,
        k=k,
        df=imdb,
        sort_algo="heap",
    )
    result = result["ID"].tolist()

    return result


# formula_1
# select + select
def pipeline_25():
    query = "How many drivers born in Europe and after 1995 have been finished the race without any disqualification and damage, and with a gap of less than 2 Laps compared with the race winner."
    answer = 2

    results = pd.read_csv("./databases/formula_1/results.csv")
    drivers = pd.read_csv("./databases/formula_1/drivers.csv")
    status = pd.read_csv("./databases/formula_1/status.csv")

    drivers = drivers[pd.to_datetime(drivers["dob"]).dt.year > 1995]
    op = LogicalSelect(operand_type=OperandType.ROW)
    drivers = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This Formula 1 driver is born in Europe.",
        df=drivers,
        depend_on=["nationality"],
    )

    op = LogicalSelect(operand_type=OperandType.ROW)
    status = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This Fomula 1 race status measn the driver finished the race without any disqualification and damage, and with a gap of less than 2 Laps compared with the race winner.",
        df=status,
        depend_on=["status"],
    )

    merged_df = pd.merge(results, drivers, on="driverId")
    merged_df = pd.merge(merged_df, status, on="statusId")

    merged_df = merged_df[["driverId"]].drop_duplicates()
    prediction = len(merged_df)

    return prediction


# formula_1
# select + order
def pipeline_26():
    query = "Rank circuits locates in East Asia and rank them based on the population of the location country in descending order. Answer with circuitId."
    answer = [17, 16, 22, 28, 35]
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")

    op = LogicalSelect(operand_type=OperandType.ROW)
    circuit = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This circuits locates in East Asia.",
        df=circuits,
        depend_on=["country"],
        thinking=True,
    )
    print(circuit.shape[0])
    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(circuit)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="population of the location country of this circuit.",
        depend_on="country",
        ascending=False,
        k=k,
        df=circuit,
        sort_algo="heap",
    )
    prediction = result["circuitId"].tolist()
    return prediction


# formula_1
# select + order
def pipeline_27():
    query = "Among constructors Lewis Hamilton has been served, which one wins the most podium in history? Answer with constructor name."
    answer = ["McLaren"]
    constructors = pd.read_csv("./databases/formula_1/constructors.csv")

    op = LogicalSelect(operand_type=OperandType.ROW)
    constructors = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Lewis Hamilton has served for this constructor.",
        df=constructors,
        depend_on=["name"],
        thinking=True,
    )
    print(constructors)

    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Count of podiums",
        depend_on="name",
        ascending=False,
        k=1,
        df=constructors,
        sort_algo="heap",
    )
    prediction = result["name"].tolist()

    return prediction


# european_football_2
# select + select
def pipeline_28():
    query = "Among Top 10 heaviest player, list players higher than basketball player Michael Jordan and has a Belgium originated name."
    answer = ["Kristof van Hout"]
    player = pd.read_csv("./databases/european_football_2/Player.csv")

    player = player.dropna(subset=["weight"])
    player = player.sort_values(by="weight", ascending=False)[:10]

    op = LogicalSelect(operand_type=OperandType.ROW)
    player = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This player is higher than Michael Jordan.",
        df=player,
        depend_on=["height"],
        thinking=True,
    )
    op = LogicalSelect(operand_type=OperandType.ROW)
    player = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This player comes from Belgium.",
        df=player,
        depend_on=["player_name"],
        thinking=True,
    )

    prediction = player["player_name"].tolist()
    return prediction


# european_football_2
# select + order
def pipeline_29():
    query = "Identify the leagues whose country is adjacent to France and is not an inland country. List the league with the longest name among them."
    answer = "Belgium Jupiler League"
    league = pd.read_csv("./databases/european_football_2/League.csv")
    country = pd.read_csv("./databases/european_football_2/Country.csv")

    op = LogicalSelect(operand_type=OperandType.ROW)
    country = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This country is adjacent to France.",
        df=country,
        depend_on=["name"],
        thinking=True,
    )
    print(country)
    op = LogicalSelect(operand_type=OperandType.ROW)
    country = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This country is not an inland country.",
        df=country,
        depend_on=["name"],
        thinking=True,
    )
    country = country[["id"]]
    merged_df = pd.merge(country, league, left_on="id", right_on="country_id")
    merged_df = LogicalOrder(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="the length of the name",
        df=merged_df,
        depend_on="name",
        k=1,
        ascending=False,
        thinking=True,
    )

    return merged_df["name"][0]


# codebase_community
# select, induce
def pipeline_30():
    query = "Among comments commented by user whose displayname is whuber on 01 Sep 2014, filter those about coding and summarize the viewpoint expressed in the comments."
    answer = """
        Commenters suggest reframing the question: rather than focusing on an R programming “fix,” the real issue is whether the statistical approach is appropriate and how such responses should be modeled. 
        They also point out that the posted code is incorrect because it does not enumerate the probabilities properly. The correct probability that two sums of digits from 0–9 match is 670/10000 ≈ 0.067, 
        which can be verified via brute force, convolution, and a Normal approximation (~0.0694).
    """
        
    users = pd.read_csv("./databases/codebase_community/users.csv")
    comments = pd.read_csv("./databases/codebase_community/comments.csv")

    users = users[users["DisplayName"] == "whuber"]
    comments = comments[
        (pd.to_datetime(comments["CreationDate"]).dt.year == 2014)
        & (pd.to_datetime(comments["CreationDate"]).dt.month == 9)
        & (pd.to_datetime(comments["CreationDate"]).dt.day == 1)
    ]

    merged_df = pd.merge(users, comments, left_on="Id", right_on="UserId")

    op = LogicalSelect(operand_type=OperandType.ROW)
    filtered_comments = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This comment is about coding.",
        df=merged_df,
        depend_on=["Text"],
        thinking=True,
    )
    filtered_comments = filtered_comments[["Text"]]
    print(filtered_comments)

    op = LogicalInduce(operand_type=OperandType.ROW)
    result = op.execute(
        ImplType.LLM_ALL,
        condition="Summarize the viewpoint expressed in the comments.",
        df=filtered_comments,
        thinking = True
    )

    return result


# codebase_community
# select + select
def pipeline_31():
    query = "Find all users in the codebase_community database whose display name starts with 'm' (case-insensitive), live in 'Vienna, Austria', and have a display name that sounds like a real human name. Return their display names."
    answer = ["Maciej Pasternacki", "Marcus Jones"]

    users = pd.read_csv("./databases/codebase_community/users.csv")
    users = users.dropna(subset=["DisplayName"])
    users["DisplayName"] = users["DisplayName"].astype(str)

    m_starters_mask = users["DisplayName"].str.lower().str.startswith("m")
    filtered_users = users[m_starters_mask]

    op = LogicalSelect(operand_type=OperandType.ROW)
    filtered_users = filtered_users[["Location", "DisplayName"]]
    filtered_users = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This user lives in Vienna, Austria.",
        df=filtered_users,
        depend_on=["Location"],
        thinking=False,
    )

    filtered_users = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This DisplayName sounds like a real human name.",
        df=filtered_users,
        depend_on=["DisplayName"],
        thinking=False,
    )
    prediction = filtered_users["DisplayName"].tolist()
    return prediction


# codebase_community
# select + impute
def pipeline_32():
    query = "Among posts with top 5 highest-Viewcount, filter those that contain external links in their post body, and extract all external links from these posts."
    answer = ['http://onlinestatbook.com/2/analysis_of_variance/one-way.html']
    posts = pd.read_csv("./databases/codebase_community/posts.csv")
    posts = posts.sort_values(by="ViewCount", ascending=False)[:5]

    op = LogicalSelect(operand_type=OperandType.ROW)
    filtered_posts = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This post body contains external link.",
        df=posts,
        depend_on=["Body"],
        thinking=True,
    )
    print(len(filtered_posts))
    print(filtered_posts['Body'])

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Extract the external links in the post body.",
        df=filtered_posts,
        depend_on="Body",
        new_col="links",
    )
    prediction = result["links"].tolist()

    return prediction


# santos
# selet + match
def pipeline_33():
    query = "Among state agencies in the ‘state_expenditures’ table with total payments exceeding 200,000 USD but less than 200,000 GBP (calculated using the peak exchange rate in 2015), list the agencies whose functions are similar to those of Canada's departments responsible for 'Agriculture and Agri-Food' as found in the ‘rm-mr-2009-eng’ table."
    answer = "[['AGRICULTURE', 'Agriculture and Agri-Food Department']]"
    state_expenditures = pd.read_csv("./databases/santos/state_expenditures.csv")
    rm_2009 = pd.read_csv("./databases/santos/rm-mr-2009-eng.csv")

    state_expenditures = state_expenditures[state_expenditures['Payments Total'] > 200000]
    print(len(state_expenditures))
    filtered_administrations = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This cost in (USD) is less than 200000 GBP, using the highest exchange rate in 2015.",
        df=state_expenditures,
        depend_on=["Payments Total"],
        thinking=True,
    )

    filtered_administrations = filtered_administrations[['Agency Name']].drop_duplicates()
    print(filtered_administrations)

    rm_2009 = rm_2009[rm_2009['MINE']=='Agriculture and Agri-Food']
    rm_2009 = rm_2009[['DEPT_EN_DESC']]

    print(rm_2009)

    op = LogicalMatch(OperandType.CELL)
    pred = op.execute(impl_type=ImplType.LLM_ONE, 
                           condition="The agency and the DEPT have similar function.", 
                           left_df = filtered_administrations, 
                           right_df = rm_2009, 
                           left_on = 'Agency Name',
                           right_on = 'DEPT_EN_DESC',
                           thinking=True)
    prediction = pred.values.tolist()
    
    return prediction

# santos
# select + match
def pipeline_34():
    query = "Among the records in the ‘dft-monthly-spend-201005’ table where the British Transport Police’s expenses on 28/05/2010 are greater than 10,000 USD, count how many have matching entries for software coding expenses on the same day in the transparency report from the ‘V3.1_-_Final__csv__May_2013__25k_Transprncy_rpt’ table."
    answer = 16
    dft_monthly_spend = pd.read_csv("./databases/santos/dft-monthly-spend-201005.csv")
    transprncy_rpt = pd.read_csv(
        "./databases/santos/V3.1_-_Final__csv__May_2013__25k_Transprncy_rpt.csv"
    )
    dft_monthly_spend = dft_monthly_spend[
        dft_monthly_spend["Entity"] == "British Transport Police"
    ]
    dft_monthly_spend = dft_monthly_spend[dft_monthly_spend["Date"] == "28/05/2010"]
    transprncy_rpt = transprncy_rpt[transprncy_rpt["Expense type"] == "Software coding"]
    transprncy_rpt = transprncy_rpt[["Date"]]
    dft_monthly_spend = dft_monthly_spend[["Date", "Amount"]]
    print(dft_monthly_spend)

    merged_df = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The amount expense in GBP is higher than 10000 USD.",
        df=dft_monthly_spend,
        depend_on=["Amount"],
        thinking=True,
    )

    print(len(merged_df), len(transprncy_rpt))

    op = LogicalMatch(OperandType.ROW)
    merged_df = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The two date shares the same day of different time.",
        left_df=merged_df,
        right_df=transprncy_rpt,
        left_on="Date",
        right_on="Date",
        thinking=True,
    )

    merged_df = merged_df.drop_duplicates()
    prediction = merged_df.shape[0]
    return prediction


# santos
# match + impute
def pipeline_35():
    query = "List the officials in ‘travel-exp-April-June-2018’ table who share the same first name with senior officials from the ‘home_office_senior_officials_travel_data_return’ table, along with their destinations and the continents where these destinations are located."
    answer = [
        ["Mark Bryson-Richardson", "Paris, France", "Europe"],
        ["Richard Montgomery", "Kathmandu/ Yangon/Manilla", "Asia"],
    ]

    home_office_travel = pd.read_csv(
        "./databases/santos/home_office_senior_officials_travel_data_return.csv"
    )
    travel_exp_2018 = pd.read_csv("./databases/santos/travel-exp-April-June-2018.csv")

    home_office_officials = home_office_travel[["Name of Official"]].drop_duplicates()
    travel_2018_officials = travel_exp_2018[
        ["Senior Officials Name", "Destination"]
    ].drop_duplicates(subset=["Senior Officials Name"])

    print(len(home_office_officials), len(travel_2018_officials))
    print(home_office_officials)
    print(travel_2018_officials)

    op = LogicalMatch(OperandType.ROW)
    merged_df = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The two senior government official names shares the same first name.",
        left_df=home_office_officials,
        right_df=travel_2018_officials,
        left_on="Name of Official",
        right_on="Senior Officials Name",
        thinking=False,
    )
    print(merged_df.columns)
    merged_df = merged_df[["right_Senior Officials Name", "right_Destination"]]

    op = LogicalImpute(operand_type=OperandType.COLUMN)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Impute the continent where this destination locates.",
        df=merged_df,
        depend_on="right_Destination",
        new_col="continent",
    )

    prediction = result.values.tolist()
    return prediction


# santos
# select + match
def pipeline_36():
    query = "From the ‘pca-human-wildlife-coexistence-animals-involved-detailed-records’ table, identify the 50 most recent wildlife incidents involving deers. For each incident, list the protected heritage area where it occurred, if that area is in a country included in the ‘cihr_co-applicant_200607’ table."
    answer = "[['Elk Island National Park of Canada', 'Canada'], ['Gros Morne National Park of Canada', 'Canada']]"

    cihr = pd.read_csv("./databases/santos/cihr_co-applicant_200607.csv")
    wildlife = pd.read_csv("./databases/santos/pca-human-wildlife-coexistence-animals-involved-detailed-records.csv")

    wildlife = wildlife.sort_values(by='Incident Number', ascending=False)[:50]
    print(wildlife['Species Common Name'])
    beers = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The species is a deer.",
        df=wildlife,
        depend_on=["Species Common Name"],
        thinking=True,
    )

    beers = beers[['Protected Heritage Area']].drop_duplicates()
    cihr= cihr[['CountryEN']].drop_duplicates()

    print(beers)
    print(cihr)

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The protected heritage area location is in the country.", 
        left_df=beers, 
        right_df=cihr, 
        left_on='Protected Heritage Area',
        right_on='CountryEN',
        thinking=True
    )
    prediction = merged_df[['left_Protected Heritage Area', 'right_CountryEN']].values.tolist()
    return prediction


# santos
# match + order
def pipeline_37():
    query = "List the matching addresses that represent the same street location in both the ‘pubs’ and ‘community_centres’ tables within the Barking and Dagenham borough, sorted by the latitude of the street location."
    answer = [["Billet Road / Rose Lane", "Rose Lane"], ["109 Bastable Avenue / Charlton Crescent", "Bastable Avenue"]]

    pubs_data = pd.read_csv("./databases/santos/pubs.csv")
    community_centres_data = pd.read_csv("./databases/santos/community_centres.csv")

    # Filter to get unique addresses from both datasets
    pubs_data = pubs_data[pubs_data["borough_name"] == "Barking and Dagenham"]
    community_centres_data = community_centres_data[
        community_centres_data["borough_name"] == "Barking and Dagenham"
    ]
    pubs_addresses = pubs_data[["address1"]].drop_duplicates()
    community_addresses = community_centres_data[["address1"]].drop_duplicates()

    print(len(pubs_addresses), len(community_addresses))
    print(pubs_addresses)
    print(community_addresses)

    op = LogicalMatch(OperandType.ROW)
    merged_df = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The two addresses represent the same street location.",
        left_df=pubs_addresses,
        right_df=community_addresses,
        left_on="address1",
        right_on="address1",
        thinking=True,
    )
    merged_df = merged_df.drop_duplicates()

    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(merged_df)
    merged_df = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="the Latitude of street.",
        depend_on="right_address1",
        ascending=True,
        k=k,
        df=merged_df,
        sort_algo="heap",
    )

    prediction = merged_df.values.tolist()
    return prediction


# santos
# match
def pipeline_38():
    query = "List all unique countries present in the CIHR co-applicant table (2006-07) for which there are protected heritage areas found in the human-wildlife coexistence table. Answer with the unique country names."
    answer = ["Canada"]

    cihr_data = pd.read_csv("./databases/santos/cihr_co-applicant_200607.csv")
    wildlife_data = pd.read_csv(
        "./databases/santos/pca-human-wildlife-coexistence-animals-involved-detailed-records.csv"
    )

    # Filter to get unique countries from both datasets
    cihr_countries = cihr_data[["CountryEN"]].drop_duplicates()
    wildlife_areas = wildlife_data[["Protected Heritage Area"]].drop_duplicates()

    print(len(cihr_countries), len(wildlife_areas))
    print(cihr_countries)
    print(wildlife_areas)

    op = LogicalMatch(OperandType.ROW)
    merged_df = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The country name matches the protected heritage area location.",
        left_df=cihr_countries,
        right_df=wildlife_areas,
        left_on="CountryEN",
        right_on="Protected Heritage Area",
        thinking=False,
    )
    merged_df = merged_df.drop_duplicates()
    # Get the number of records from the left table after LogicalMap
    prediction = merged_df["left_CountryEN"].drop_duplicates().tolist()
    return prediction


# santos
# match + select
def pipeline_39():
    query = "List suppliers that appear in both the '01.Apr_2018' table and the '2015_05_expenditure' table that represent the same entity, and offer professional medical supplies."
    answer = "[['Nhs Supply Chain', 'NHS SUPPLY CHAIN'], ['Novartis Pharmaceuticals Uk Ltd', 'NOVARTIS PHARMACEUTICALS UK LTD'], ['Philips Healthcare', 'PHILIPS HEALTHCARE'], ['Nhs Blood And Transplant', 'NHS BLOOD & TRANSPLANT']]"
    left = pd.read_csv("./databases/santos/01.Apr_2018.csv")
    right = pd.read_csv("./databases/santos/2015_05_expenditure.csv")
    left = left[['Supplier']].drop_duplicates()
    right = right[['Supplier']].drop_duplicates()
    print(len(left), len(right))

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_ONE, 
                           condition="The two suppliers are the same supplier.", 
                           left_df=left, 
                           right_df=right, 
                           left_on='Supplier',
                           right_on='Supplier',
                           thinking=True)

    merged_df = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The company is related to professional medical supplies.",
        df=merged_df,
        depend_on="left_Supplier",
    )
    prediction = merged_df.values.tolist()
    return prediction


# nextiaJD
# match
def pipeline_40():
    query = "List the unique addresses from the 'cultural-spaces' table that have matching site addresses in the 'public-art' table."
    answer = ["2025 W 11th Av, Vancouver, BC, V6J 2C7"] 

    cultural_spaces = pd.read_csv("./databases/nextiaJD/cultural-spaces.csv")
    public_art = pd.read_csv("./databases/nextiaJD/public-art.csv")

    cultural_addresses = cultural_spaces[["ADDRESS"]].drop_duplicates()
    art_site_addresses = public_art[["SiteAddress"]].drop_duplicates()

    print(len(cultural_addresses), len(art_site_addresses))
    print(cultural_addresses)
    print(art_site_addresses)

    op = LogicalMatch(OperandType.ROW)
    merged_df = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The two addresses represent the same location.",
        left_df=cultural_addresses,
        right_df=art_site_addresses,
        left_on="ADDRESS",
        right_on="SiteAddress",
        thinking=False,
    )
    merged_df = merged_df.drop_duplicates()
    prediction = merged_df[['ADDRESS']].drop_duplicates().values.tolist()
    return prediction


# mondial_geo
# select + match
def pipeline_41():
    query = "For each Asian city with more than 5 million people, what is the closest sea?"
    answer = "[['Bangkok', 'Pacific Ocean'], ['Bangkok', 'South China Sea'], ['Beijing', 'Pacific Ocean'], ['Beijing', 'Yellow Sea'], ['Hong Kong', 'Gulf of Mexico'], ['Hong Kong', 'Pacific Ocean'], ['Hong Kong', 'South China Sea'], ['Istanbul', 'Gulf of Aden'], ['Jakarta', 'Sunda Sea'], ['Karachi', 'Arabian Sea'], ['Lahore', 'Arabian Sea'], ['Lahore', 'North Sea'], ['Lahore', 'Red Sea'], ['Mumbai', 'Arabian Sea'], ['New Delhi', 'Bay of Bengal'], ['Seoul', 'Sea of Japan'], ['Seoul', 'Yellow Sea'], ['Shanghai', 'East China Sea'], ['Shanghai', 'Kattegat'], ['Shanghai', 'Yellow Sea'], ['Tianjin', 'Yellow Sea'], ['Tokyo', 'Pacific Ocean'], ['Tokyo', 'Sea of Japan']]"

    city = pd.read_csv("./databases/mondial_geo/city.csv")
    city = city[city['Population'] > 5000000]
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    print(len(city))

    city = LogicalSelect(OperandType.ROW).execute(
        impl_type= ImplType.LLM_ONE,
        condition="The city locates in an Asian Country.",
        df=city,
        depend_on=["Name","Country","Province"],
        thinking = True
    )
    print(len(city))
    print(len(sea))

    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_ONE,
                    condition="Among all provided seas, map each city to its closest sea.", 
                    left_df=city, 
                    right_df=sea, 
                    left_on='Name',
                    right_on='Name',
                    thinking=True)

    prediction = merged_df[['left_Name', 'right_Name']].values.tolist()
    return prediction


# mondial_geo
# order + match
def pipeline_42():
    query = "List the continents that adjoin each of the world's five largest seas."
    answer = "[['Pacific Ocean', 'America'], ['Pacific Ocean', 'Asia'], ['Pacific Ocean', 'Australia/Oceania'], ['Atlantic Ocean', 'Africa'], ['Atlantic Ocean', 'America'], ['Atlantic Ocean', 'Europe'], ['Indian Ocean', 'Africa'], ['Indian Ocean', 'Asia'], ['Indian Ocean', 'Australia/Oceania'], ['Arctic Ocean', 'America'], ['Arctic Ocean', 'Asia'], ['Arctic Ocean', 'Europe'], ['Arabian Sea', 'Asia']]"
   
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    continent = pd.read_csv("./databases/mondial_geo/continent.csv")

    sea = LogicalOrder(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The area of the sea.",
        df=sea,
        depend_on="Name",
        k=5,
        ascending=False,
        sort_algo="heap",
        thinking = True
    )

    print(sea)
    print(continent)

    # 36*5
    op = LogicalMatch(OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_ONE,
                           condition="The sea adjoins the continent.", 
                           left_df=sea, 
                           right_df=continent, 
                           left_on='Name',
                           right_on='Name',
                           thinking=True)
    prediction = merged_df[['left_Name', 'right_Name']].values.tolist()
    return prediction


# disney
# select + match
def pipeline_43():
    query = "Calculate the average total revenue for years in which musical movies were released after the Beijing Summer Olympics, by matching movie release dates with revenue data from the same year."
    answer = [[48813.0]]
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")
    revenue = pd.read_csv("./databases/disney/revenue.csv")

    filtered_revenue = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This year is later than the year of Beijing Summer Olympics.",
        df=revenue,
        depend_on=["Year"],
        thinking=True,
    )

    print(filtered_revenue)

    movies_total_gross = movies_total_gross[movies_total_gross["genre"] == "Musical"]
    print(len(filtered_revenue), len(movies_total_gross))

    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="The two date are in the same year.",
        left_df=movies_total_gross,
        right_df=filtered_revenue,
        left_on="release_date",
        right_on="Year",
        thinking=True,
    )

    prediction = merged_df.agg(avgTotalRevenue=("right_Total", "mean"))

    return prediction.values.tolist()


# disney
# select + impute
def pipeline_44():
    query = "Calculate the average profit margin (total gross divided by inflation-adjusted gross) for musical movies that were released after the end of World War II."
    answer = 0.5298
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")

    # Filter movies to only include comedy genre
    movies_total_gross = movies_total_gross[movies_total_gross["genre"] == "Musical"]
    print(len(movies_total_gross))
    print(movies_total_gross)

    movies_total_gross = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This date is after end of World War II.",
        df=movies_total_gross,
        depend_on=["release_date"],
        thinking=True,
    )

    # Calculate profit margin using LogicalImpute
    movies_total_gross = LogicalImpute(operand_type=OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Calculate the profit margin as total gross divided by inflation-adjusted gross.",
        df=movies_total_gross,
        depend_on=["total_gross", "inflation_adjusted_gross"],
        new_col="profit_margin",
        thinking=True,
    )

    # Calculate average profit margin (convert to int first)
    prediction = movies_total_gross["profit_margin"].astype(float).mean()
    print(prediction)

    return prediction


# disney
# select + cluster
def pipeline_45():
    query = "Group adventure movies released after the release date of the movie 'Transformers' directed by Michael Bay by MPAA rating into two categories (suitable for children vs not suitable for children), and count the number of movies in each category."
    answer = [['Not Suitable for Children', 9], ['Suitable for Children', 39]]
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")

    # Filter movies to only include adventure genre
    movies_total_gross = movies_total_gross[movies_total_gross["genre"] == "Adventure"]
    print(len(movies_total_gross))

    # Filter movies released after 1980
    movies_total_gross = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This movie was released after the release date of movie 'Transformers' directed by Michale Bay.",
        df=movies_total_gross,
        depend_on=["release_date"],
        thinking=True,
    )

    print(len(movies_total_gross))

    # Group by MPAA rating and calculate average total gross
    grouped_result = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Group by MPAA rating into two levels, including 'not suitable for children' and 'suitable for children'.",
        df=movies_total_gross,
        depend_on=["MPAA_rating"],
        thinking=True,
    )

    grouped_result = (
        grouped_result[["cluster_name", "movie_title"]]
        .groupby("cluster_name")
        .agg(avgTotalRevenue=("movie_title", "count"))
        .reset_index()
    )

    prediction = grouped_result.values.tolist()
    print(prediction)

    return prediction


# disney
# select, induce
def pipeline_46():
    query = "Identify the top 3 highest-grossing movies directed by Disney directors born outside of the United States and summarize the filmmaking styles reflected by these movies."
    answer = """
        Overall, these PG-rated adventures blend humor, heart, and richly realized worlds, appealing to a broad family audience and delivering strong box-office performance.
        
        - Wolfgang Reitherman (The Jungle Book): Classic Disney hand-drawn style paired with a breezy, episodic structure. Character animation and musical set pieces carry the narrative, while jazzy songs, a charismatic animal ensemble, playful villainy, and clear, economical staging create a warm, relaxed tone.
        - Byron Howard (Zootopia): Slick CG with snappy buddy-cop pacing and a neon-noir flavor. Meticulous world-building uses species traits to drive both visual gags and plot logic; kinetic chases, sharp banter, and an accessible allegory of bias and belonging underpin the comedy.
        - Don Hall (Big Hero 6): Polished CG that fuses superhero spectacle with heartfelt family drama. A soft, huggable design language (Baymax) sits within a vibrant hybrid cityscape; readable, gadget-driven action, themes of grief, healing, and found family, and gentle humor anchor the set pieces.
    """
    
    director = pd.read_csv("./databases/disney/director.csv")
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")

    filtered_director = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This director was born outside of the United States.",
        df=director,
        depend_on=["name"],
        thinking=True,
    )

    print(len(filtered_director))
    print(filtered_director)
    movies_total_gross["total_gross"] = pd.to_numeric(
        movies_total_gross["total_gross"].str.replace(r"[^\d.-]", "", regex=True),
        errors="coerce",
    ).astype("Int64")
    merged_df = pd.merge(
        filtered_director, movies_total_gross, left_on="name", right_on="movie_title"
    )

    merged_df = merged_df.sort_values(by="total_gross", ascending=False)[:3]

    print(merged_df)
    op = LogicalInduce(operand_type=OperandType.ROW)
    prediction = op.execute(
        ImplType.LLM_ALL,
        condition="Summarize the filmmaking style of the movies directed by those directors",
        df=merged_df,
        thinking=True,
    )

    return prediction


# disney
# select + order
def pipeline_47():
    query = "List the directors of Disney adventure movies released after the dissolution of the Soviet Union and sort them by age in ascending order."
    answer = [["Nathan Greno"],["Stephen J. Anderson"],["Don Hall"],["Byron Howard"],["Ralph Zondag"],["Rich Moore"],["Chris Sanders"],["Robert Walker"],["Gary Trousdale"],["Mark Dindal"],["Barry Cook"],["Chris Buck"],["Mike Gabriel"],["Ron Clements"],["Roger Allers"],["Wolfgang Reitherman"],["Clyde Geronimi"]]
    director = pd.read_csv("./databases/disney/director.csv")
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")

    # Filter characters from movies released in the 1990s
    movies_total_gross = movies_total_gross[movies_total_gross["genre"] == "Adventure"]

    movies_total_gross = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This movie was released after Dissolution of the Soviet Union.",
        df=movies_total_gross,
        depend_on=["release_date"],
        thinking=True,
    )

    print(movies_total_gross)

    merged_df = pd.merge(
        director, movies_total_gross, left_on="name", right_on="movie_title"
    )
    print(merged_df)

    merged_df = merged_df[["director"]].drop_duplicates()

    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(merged_df)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="Age of these Disney movie directors.",
        depend_on="director",
        ascending=True,
        k=k,
        df=merged_df,
        sort_algo="heap",
        thinking=True,
    )

    prediction = result.values.tolist()

    return prediction

# car_retails
# groupby, induce
def pipeline_48():
    query = "Group customers by continent (inferred from their country), join each customer's orders with the corresponding products, and generate a one-sentence summary of car retail preferences for each continent. Return a list of [continent, preference summary] pairs."
    answer = """
    Asian customers: Asia’s car retail preferences in this dataset skew strongly toward collectible, high-emotion vehicles rather than everyday commuters. 
    Key patterns: 
        1) American muscle and classics lead: repeated interest in 1960s–1970s icons such as Dodge Charger (’68/’69), Ford Mustang (’68), Plymouth Hemi Cuda (’70), Chevy Chevelle SS 454 (’70), Camaro Z28 (’69/’82), and Corvettes (’58, ’02). Vintage pickups and hot rods (’40 Ford Pickup, ’57 Chevy Pickup, Ford Model A/J-Coupe, ’34 Ford V8) also resonate. 
        2) European exotics and historic icons are strong: Ferrari Enzo and 360 Spider, Aston Martin DB5, Mercedes-Benz 500K Roadster, Jaguar XK120, Alfa Romeo 8C 2300, multiple Porsche 356 variants, Alpine Renault, and Lancia Delta 16V recur. 
        3) Nostalgia/retro draws: VW Microbus (’62) appears multiple times; Chrysler Plymouth Prowler also features. 
        4) Japanese performance/tuner niche: Mazda RX-7 (’93) and Honda Civic (’95) present but less frequent than US/EU classics. 
        5) Era mix: concentration in 1950s–1970s, with notable pre-war collectibles (Ford Model A/T, Buick Runabout) and a few 1990s–2000s halo cars. 
    Overall, demand favors classic/vintage and performance-oriented models with strong brand heritage and nostalgia appeal; modern mass-market sedans are underrepresented.
    
    European customers: European customers show a strong collector/performance bias in car retail. 
    Preferences cluster around: 
        1) Classic/performance icons from the 1960s–1970s, especially American muscle—1968 Ford Mustang, 1969 Dodge Charger and Super Bee, 1970 Plymouth Hemi Cuda, 1970 Chevy Chevelle SS 454, 1969 Camaro Z28, 1957/2002 Corvettes, 1961 Impala, 1976 Gran Torino. 
        2) European sports/exotics and heritage models—Aston Martin DB5, Porsche 356 (Roadster/Coupe) and other Porsches, Ferrari 360 Spider and Enzo, Jaguar XK120, Alfa Romeo GTA/8C 2300, Mercedes-Benz SSK/500K, Alpine Renault 1600s, Citroën 15CV, Lancia Delta 16V. 
        3) Pre-war and early classics—Ford Model T/Model A variants, Buick Runabout/Marquette Phaeton, Chrysler Airflow, Cadillac and Horch limousines, Lincoln Berline—indicating strong interest in vintage/coachbuilt pieces. 
        4) Modern mass-market is minimal (e.g., 1995 Honda Civic), with occasional Japanese sports (Mazda RX-7). 
        5) Some exposure to utility/people-movers (VW Microbus, Greyhound/Setra/Mercedes buses, pickups/trucks), but these appear secondary and likely collected for nostalgia rather than daily use. 
    Overall, Europe’s demand skews heavily toward collectible, historically significant, and performance-oriented cars—both European marques and American muscle—over contemporary mainstream models or SUVs.
    
    North American customers: customers' preferences in North America skew strongly toward collectible and performance-oriented cars with a heavy nostalgia factor.
        1) Era focus: Dominated by mid‑century to 1970s classics, with peaks in late‑1960s/early‑1970s muscle (1969 Camaro Z28; 1970 Chevelle SS 454; 1968 Mustang; 1968–69 Dodge Charger; 1970 Plymouth Hemi Cuda) and 1950s icons (1957 Corvette, 1957 Ford Thunderbird). Pre‑war and early‑vintage pieces also recur (Ford Model T/Model A, 1904 Buick Runabout, 1917 Maxwell, 1934–39 Fords/Cadillacs).
        2) Brand/origin bias: Strong American bias (Chevrolet, Ford, Dodge/Plymouth) complemented by high‑interest European collectibles and exotics—especially Porsche 356 variants (1948/1956), Mercedes 500K/SSK, Jaguar XK120, Alfa Romeo (8C, GTA), and halo cars (Ferrari 360 Spider, Ferrari Enzo, Lamborghini Diablo).
        3) Body styles and themes: Coupes and roadsters/convertibles dominate, signaling a preference for sporty silhouettes and open‑top classics. Limited editions and commemoratives appear (e.g., 1958 Chevy Corvette Limited Edition, Indy 500 Monte Carlo SS), reinforcing collector appeal.
        4) Trucks/pickups crossover: Notable demand for vintage light trucks alongside cars (1957 Chevy Pickup; 1940/1940s Ford trucks), indicating affinity for Americana and utility nostalgia.
        5) Modern mass‑market/tuner presence: Secondary interest in 1990s–2000s imports (Honda Civic 1995; Mazda RX‑7 1993; Porsche Cayenne Turbo), much less frequent than classics/exotics.
    Summary: North American customers prioritize iconic American muscle and classic sports/convertible models, with strong secondary interest in European pre‑war luxury and post‑war exotics, plus a meaningful niche for vintage pickups—overall signaling a collector‑driven, performance‑and‑nostalgia market profile.
    
    Oceanian customers: Oceania customers show a clear preference for collectible and performance-oriented cars rather than everyday modern models.
        1) Dominant interest: American classics and muscle (1950s–1970s). Frequent models include 1968 Ford Mustang, 1969 Chevrolet Camaro Z28, 1970 Chevy Chevelle SS 454, 1970 Plymouth Hemi Cuda, 1969/1968 Dodge Charger, 1969 Dodge Super Bee, 1957 Ford Thunderbird, 1957 Corvette Convertible, 1961 Chevrolet Impala, 1976 Ford Gran Torino. Early American icons (1903 Ford Model A, 1932 Model A Ford J-Coupe, 1934 Ford V8 Coupe) also appear, underscoring nostalgia.
        2) Strong secondary interest: European heritage sports and prewar luxury. Repeated appearances of 1936 Mercedes-Benz 500K/500k and 1928 Mercedes-Benz SSK, Porsche 356 variants, 1949 Jaguar XK 120, 1932 Alfa Romeo 8C2300, 1971 Alpine Renault 1600s, 1965 Aston Martin DB5. Modern exotics (Ferrari Enzo, Ferrari 360 Spider, Lamborghini Diablo) are present but fewer.
        3) Tertiary interest: Japanese tuner/compact cars (1993 Mazda RX-7, 1995 Honda Civic) indicate some appetite for imports, but far less than American/European classics.
        4) Body styles: Coupes and convertibles dominate; sedans/hatchbacks are rare. SUVs are essentially absent (only a lone Porsche Cayenne appears).
    Implication for car retail: Stock and market toward classic/muscle American nameplates and European vintage/exotics; emphasize performance, authenticity, and restoration-friendly offerings. Maintain a smaller selection of Japanese enthusiast models to capture niche demand.
    """
    customer = pd.read_csv("./databases/car_retails/customers.csv")
    order = pd.read_csv("./databases/car_retails/orders.csv")
    order_details = pd.read_csv("./databases/car_retails/orderdetails.csv")
    product = pd.read_csv("./databases/car_retails/products.csv")

    grouped_result = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Group by the continent where the country locates.",
        df=customer,
        depend_on=["country"],
        thinking=True,
    )
    print(grouped_result)

    merged_df = pd.merge(grouped_result, order, on="customerNumber")
    merged_df = pd.merge(merged_df, order_details, on="orderNumber")
    merged_df = pd.merge(merged_df, product, on="productCode")
    merged_df = merged_df[["productName", "cluster_name"]]
    
    print(merged_df)
    print(len(merged_df))

    group_results = []
    for cluster_name, grp in merged_df.groupby("cluster_name", as_index=False):
        summary = LogicalInduce(operand_type=OperandType.ROW).execute(
            impl_type=ImplType.LLM_ALL,
            condition="Summarize the preference of customers in this continent on car retails.",
            df=grp,
            thinking=True,
        )
        group_results.append([cluster_name, summary])

    result = pd.DataFrame(group_results, columns=["cluster_name", "preference"])

    prediction = result.values.tolist()
    return prediction


# car_retails
# select, groupby
def pipeline_49():
    query = "For all shipped orders with non-empty comments that requested DHL shipping, group the customers' living locations into three categories: 'Urban', 'Rural', or 'Unknown', and for each category, count the number of such customers."
    answer = [["Rural", 1], ["Urban", 5]]
    order = pd.read_csv("./databases/car_retails/orders.csv")
    customer = pd.read_csv("./databases/car_retails/customers.csv")

    order = order[order["status"] == "Shipped"].dropna(subset=("comments"))
    order = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This commnet asked for DHL shipping.",
        df=order,
        depend_on=["comments"],
        thinking=True,
    )
    merged_df = pd.merge(order, customer, on="customerNumber")

    print(merged_df)

    grouped_result = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Cluster thses locations into three buckets named 'Urban', 'Rural' or 'Unkown'.",
        df=merged_df,
        depend_on=[
            "addressLine1",
            "addressLine2",
            "city",
            "state",
            "postalCode",
            "country",
        ],
        thinking=True,
    )

    grouped_result = grouped_result[['customerNumber','cluster_name']].drop_duplicates()

    grouped_result = (
        grouped_result.groupby("cluster_name")
        .agg(avgTotalRevenue=("customerNumber", "count"))
        .reset_index()
    )

    prediction = grouped_result.values.tolist()
    return prediction


# mondial_geo
# select + groupby
def pipeline_50():
    query = "Among European countries with a population greater than one-thirtieth of China's population, group them by level of economic development and count the number of countries in each group."
    answer = [["Developed", 4], ["Developing", 1], ["Emerging", 2]]

    country = pd.read_csv("./databases/mondial_geo/country.csv")
    encompasses = pd.read_csv("./databases/mondial_geo/encompasses.csv")

    european_countries = encompasses[encompasses["Continent"] == "Europe"][
        "Country"
    ].tolist()
    european_data = country[country["Code"].isin(european_countries)]

    filtered_countries = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This country has population over one-thirtieth of China's population.",
        df=european_data,
        depend_on=["Population"],
        thinking=True,
    )
    print(filtered_countries)

    grouped_result = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Group these countries by their economic development level into categories like 'Developed', 'Developing', and 'Emerging'.",
        df=filtered_countries,
        depend_on=["Name", "Population", "Area"],
        thinking=True,
    )

    result = (
        grouped_result.groupby("cluster_name")
        .agg(country_count=("Name", "count"))
        .reset_index()
    )

    prediction = result.values.tolist()
    return prediction


# music
# select + groupby
def pipeline_51():
    query = "For Eminem's albums released after 2000 that contain love-related singles, group them by decade and count the number of albums in each decade."
    answer = [["2000s", 3], ["2010s", 3]]  

    amazon_music = pd.read_csv("./databases/music/amazon_music.csv")

    albums = amazon_music[amazon_music["Artist_Name"] == "Eminem"]
    albums["Released"] = pd.to_datetime(albums["Released"])
    filtered_albums = albums[albums["Released"].dt.year > 2000]

    print(len(albums))

    filtered_albums = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This album contains love-related singles.",
        df=filtered_albums,
        depend_on=["Album_Name", "Song_Name"],
        thinking=True,
    )
    filtered_albums = filtered_albums.drop_duplicates(subset=["Album_Name"])
    print(len(filtered_albums))

    grouped_albums = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Group these albums by decade (2000s, 2010s, 2020s).",
        df=filtered_albums,
        depend_on=["Released"],
        thinking=True,
    )

    result = (
        grouped_albums.groupby("cluster_name")
        .agg(album_count=("Album_Name", "count"))
        .reset_index()
    )

    prediction = result.values.tolist()
    return prediction


# formula_1
# select + match
def pipeline_52():
    query = "Which circuits located in Europe are named after a constructor originating from Europe? Answer with the circuit name."
    answer = ['Circuit de Barcelona-Catalunya', 'Circuit de Nevers Magny-Cours', 'Hockenheimring', 'Circuit de Spa-Francorchamps', 'Autodromo Nazionale di Monza', 'Nürburgring', 'Autodromo Enzo e Dino Ferrari', 'A1-Ring', 'Circuit Paul Ricard', 'Brands Hatch', 'Dijon-Prenois', 'Charade Circuit', 'Rouen-Les-Essarts', 'Le Mans', 'Reims-Gueux', 'Zeltweg', 'Circuit Bremgarten', 'Circuit de Pedralbes', 'Red Bull Ring']
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    constructors = pd.read_csv("./databases/formula_1/constructors.csv")

    circuits = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The country is in Europe",
        df=circuits,
        depend_on="country",
        thinking = True,
    )
    print(circuits.shape[0])  # 37
    constructors = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The nationality corresponds to an European country",
        df=constructors,
        depend_on="nationality",
        thinking = True,
    )
    print(constructors.shape[0])  # 150
    merged = LogicalMatch(operand_type=OperandType.CELL).execute(
        impl_type=ImplType.LLM_SEMI,
        condition="The circuit name is named after the provided constructor name",
        left_df=circuits,
        right_df=constructors,
        left_on="name",
        right_on="name",
        thinking = True,
    )
    print(merged)
    return merged["left_name"].tolist()


# beer_factory
# select + cluster
def pipeline_53():
    query = "Among brands that receive positive comments, cluster them by their locations into 'East', 'West', and 'Central' regions, based on the geographical location of their origin with respect to the United States."
    answer = [["Captain Eli's", 'East'], ["Fitz's", 'Central'], ['Sprecher', 'Central'], ['Bulldog', 'West'], ["Sparky's Fresh Draft", 'West'], ['Gales', 'Central'], ['Bundaberg', 'Non-US']]
    beer_brand = pd.read_csv("./databases/beer_factory/rootbeerbrand.csv")
    beer_review = pd.read_csv("./databases/beer_factory/rootbeerreview.csv")

    beer_review = beer_review.dropna(subset=["Review"])
    filtered_review = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This comment is positive.",
        df=beer_review,
        depend_on=["Review"],
        thinking=True,
    )

    print(len(filtered_review))

    merged_df = pd.merge(filtered_review, beer_brand, on="BrandID")

    grouped_brand = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Cluster by the US state into 'East', 'West' and 'Central' based on geographical location.",
        df=merged_df,
        depend_on=["State"],
        thinking=True,
    )

    print(grouped_brand.columns)
    prediction = grouped_brand[["BrandName", "cluster_name"]].values.tolist()
    return prediction


# beer_factory
# select, induce
def pipeline_54():
    query = "Summarize characteristics of those customers who have bought beers that originate from the Great Lakes Region."
    answer = """
    Overview of customer characteristics:
    
    - Geography: All customers are in California, overwhelmingly concentrated in Sacramento. Notable nearby clusters include Elk Grove, Citrus Heights, Carmichael, Folsom, Rancho Cordova, North Highlands, Antelope, Orangevale, Rio Linda, Fair Oaks, Rancho Murieta, Mather, Wilton, Herald, Galt, Isleton, Courtland, Walnut Grove, and Ryde.
    - Zip codes: Heavily clustered in Sacramento’s 942xx and 958xx ranges, typical of central-city and metro-area neighborhoods.
    - Contact details: Phone numbers are predominantly in the 916 area code (Greater Sacramento). A minority use 209 (e.g., Herald/Galt). Email domains span the major providers—Gmail, Outlook/Hotmail, iCloud, Yahoo, AOL, Fastmail, and Zoho—no single domain dominates.
    - First purchase timing: Records span from 2012-07-02 to 2013-09-18, with most first purchases occurring during 2013 (steady activity from January through September).
    - Email subscription: Both opted-in and opted-out customers are well represented; there is a substantial reachable audience via email, but many have not subscribed.
    - Gender: Mixed and fairly balanced representation of male and female customers.
    - Other notes: The data appear primarily residential addresses. Some surnames recur (e.g., family members or common names), but entries typically have distinct addresses/emails.
    
    Implications: The customer base is strongly local to the Sacramento metro area, with communications best tailored to 916-area contacts and common consumer email providers. Engagement efforts in 2013 captured many first-time customers, suggesting effective acquisition during that period; ongoing opt-in growth could expand reach.
    """
    customer = pd.read_csv("./databases/beer_factory/customers.csv")
    beer_brand = pd.read_csv("./databases/beer_factory/rootbeerbrand.csv")
    transaction = pd.read_csv("./databases/beer_factory/transaction.csv")
    beer = pd.read_csv("./databases/beer_factory/rootbeer.csv")

    customer = customer.rename(columns = {"City":"customer_City", "State": "customer_State"})
    
    filtered_brand = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This state is in the Great Lakes Region.",
        df=beer_brand,
        depend_on=["State", "Country"],
        thinking=True,
    )
    print(len(filtered_brand)) # 7

    merged_df = pd.merge(customer, transaction, on="CustomerID")
    merged_df = pd.merge(merged_df, beer, on="RootBeerID")
    merged_df = pd.merge(merged_df, filtered_brand, on="BrandID")
    merged_df = merged_df[["First","Last","StreetAddress","customer_City","customer_State","ZipCode","Email","PhoneNumber","FirstPurchaseDate","SubscribedToEmailList","Gender"]].drop_duplicates()
    print(len(merged_df)) #499
    
    prediction = LogicalInduce(operand_type=OperandType.ROW).execute(
        ImplType.LLM_ALL,
        condition="Summarize characteristics of those customers",
        df=merged_df,
        thinking = True,
    )

    return prediction


# beer_factory
# select + cluster
def pipeline_55():
    query = "Among root beer brands that began their first brew more than 50 years before the Beijing Summer Olympics, cluster them by their location's Köppen–Geiger climate types, then count how many brands belong to each climate cluster."
    answer = [['Hot-summer Mediterranean (Csa)', 1], ['Hot-summer humid continental (Dfa)', 5], ['Humid subtropical (Cfa)', 3], ['Warm-summer Mediterranean (Csb)', 2]]
    beer_brand = pd.read_csv("./databases/beer_factory/rootbeerbrand.csv")
    print(len(beer_brand))
    
    filtered_brand = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This year is more than 50 years before the Beijing Summer Olympics.",
        df=beer_brand,
        depend_on=["FirstBrewedYear"],
        thinking=True,
    )

    # print(len(filtered_brand))
    print(filtered_brand)

    grouped_brands = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ALL,
        condition="Cluster these locations by their Köppen–Geiger climate types ",
        df=filtered_brand,
        depend_on=["City", "State", "Country"],
        thinking=True,
    )
    
    result = (
        grouped_brands.groupby("cluster_name")
        .agg(brand_count=("BrandName", "count"))
        .reset_index()
    )
    
    prediction = result.values.tolist()
    return prediction


# beer_factory
# select + order
def pipeline_56():
    query = "Find all male customers with Outlook email addresses who live in Sacramento (the capital of California). Sort them by the latitude of their address, from highest to lowest, and answer with their first and last names."
    answer = [['Dominic', 'Sandoval'], ['Thomas', 'Kelly'], ['Derrick', 'Lewis'], ['Trevor', 'Matich'], ['Scott', 'Boras'], ['Gary', 'Hoffman'], ['Bob', 'Wilkins'], ['Onterrio', 'Smith'], ['Frenchy', 'Bordagaray'], ['Lance', 'Briggs'], ['Jackie', 'Greene'], ['Scott', 'Galbraith'], ['Bob', 'Elliott'], ['Raphael', 'Saadiq'], ['Dion', 'James'], ['Rush', 'Limbaugh'], ['Michael', 'Roe'], ['John', 'Bowker'], ['Randy', 'Veres'], ['Jim', 'Loscutoff'], ['Max', 'Baer']]
    customer = pd.read_csv("./databases/beer_factory/customers.csv")

    customer = customer[customer['Email'].str.endswith('@outlook.com')]
    customer = customer[customer['Gender'] == 'M']
    filtered_customer = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This City is the capital of California.",
        df=customer,
        depend_on=["City"],
        thinking=True,
    )

    print(len(filtered_customer))

    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(filtered_customer)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="the latitude of this location",
        depend_on=['StreetAddress','City','State'],
        ascending=False,
        k=k,
        df=filtered_customer,
        sort_algo="heap",
        thinking = True,
    )

    print(result)
    prediction = result[['First','Last']].values.tolist()

    return prediction


# menu
# select + cluster
def pipeline_57():
    query = "Consider the spicy dishes that appear in menus from New York restaurants after 1970. Group these dishes by their countries of origin, and count how many dishes belong to each cluster."
    answer = [['Germany', 1], ['India', 2], ['Italy', 2], ['United States', 1]]

    menu = pd.read_csv("./databases/menu/Menu.csv")
    dish = pd.read_csv("./databases/menu/Dish.csv")
    menu_item = pd.read_csv("./databases/menu/MenuItem.csv")
    menu_page = pd.read_csv("./databases/menu/MenuPage.csv")

    menu = menu.rename(columns = {"id":"menu_id"})
    menu_page = menu_page.rename(columns = {"id":"menu_page_id"})
    dish = dish.rename(columns = {"id":"dish_id", "name": "dish_name"})

    # Filter menus from New York after 1950
    menu["date"] = pd.to_datetime(menu["date"], errors="coerce")
    ny_menus = menu[
        (menu["place"].str.contains("NEW YORK", case=False, na=False)) & 
        (menu["date"].dt.year > 1970)
    ]
    print(len(ny_menus))


    # Join tables to get dishes from these menus
    merged_df = pd.merge(ny_menus, menu_page, on = "menu_id")
    merged_df = pd.merge(merged_df, menu_item, on="menu_page_id")
    merged_df = pd.merge(merged_df, dish, on="dish_id")

    merged_df = merged_df[['dish_name', 'description']].drop_duplicates()
    print(len(merged_df))
    
    
    filtered_dishes = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This dish is spicy.",
        df=merged_df,
        depend_on=["dish_name", "description"],
        thinking=True,
    )

    # print(filtered_dishes[["dish_name", "description"]])
    
    grouped_dishes = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Group these dishes by their country of origin.",
        df=filtered_dishes,
        depend_on=["dish_name", "description"],
        thinking=True,
    )
    
    result = (
        grouped_dishes.groupby("cluster_name")
        .agg(dish_count=("dish_name", "count"))
        .reset_index()
    )
    
    prediction = result.values.tolist()
    return prediction


# menu
# select + order
def pipeline_58():
    query = "Find all expensive dishes (lowest price > $200) that appeared in hotel menus, then sort them by the time they first appeared from earliest to latest. Answer with the dish names and the year each dish first appeared."
    answer = [['Beef or Ham Sandwich with BBQ Sauce, beef or ham a plenty, french fries, tomato garnish on a french roll', 0], ["Remy Martin 'Lous XIII'", 0], ['Pintadon en cocotte Fermiere', 1906], ['Special raised Squab Turkey', 1912], ['squab duckling ', 1912], ['special raised squab Turkey', 1912], ['Friend Georgia Fantail Shrimps, Sauce Tartare , Potato Chips', 1959], ['Chateau Latour, 1955, Pauillac', 1986]] 
    
    menu = pd.read_csv("./databases/menu/Menu.csv")
    dish = pd.read_csv("./databases/menu/Dish.csv")
    menu_item = pd.read_csv("./databases/menu/MenuItem.csv")
    menu_page = pd.read_csv("./databases/menu/MenuPage.csv")


    menu = menu.rename(columns = {"id":"menu_id"})
    menu_page = menu_page.rename(columns = {"id":"menu_page_id"})
    dish = dish.rename(columns = {"id":"dish_id", "name": "dish_name"})

    expensive_dishes = dish[dish["lowest_price"] > 200.0]
    expensive_dishes = expensive_dishes.drop_duplicates(subset=["dish_name"])

    merged_df = pd.merge(menu, menu_page, on = "menu_id")
    merged_df = pd.merge(merged_df, menu_item, on="menu_page_id")
    merged_df = pd.merge(merged_df, expensive_dishes, on="dish_id")
    
    print(len(merged_df))

    
    # Filter hotel menus
    hotel_menus = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This menu is from a hotel restaurant.",
        df=merged_df,
        depend_on=["venue", "sponsor", "place"],
        thinking=True,
    )
    print(len(hotel_menus))

    # Sort by first appearance date
    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(hotel_menus)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="the first appearance time of this dish",
        depend_on=["first_appeared"],
        ascending=True,
        k=k,
        df=hotel_menus,
        sort_algo="heap",
        thinking=True,
    )
    
    prediction = result[["dish_name", "first_appeared"]].values.tolist()
    return prediction


# menu
# select + cluster
def pipeline_59():
    query = "Consider dishes that contain seafood ingredients and have a lowest price greater than 100 USD. Cluster them by their preparation methods into 'Baked', 'Grilled', 'Fried', 'Raw', 'Sautéed', 'Steamed' and 'Others'. Count how many dishes use each preparation method."
    answer = [['Baked', 4], ['Fried', 2], ['Grilled', 2], ['Others', 10], ['Raw', 3], ['Sautéed', 3], ['Steamed', 4]]
    dish = pd.read_csv("./databases/menu/Dish.csv")

    expensive_dishes = dish[dish["lowest_price"] > 100.0]
    expensive_dishes = expensive_dishes.drop_duplicates(subset=["name"])

    # Filter seafood dishes
    seafood_dishes = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This dish contains seafood ingredients (fish, shrimp, lobster, crab, etc.).",
        df=expensive_dishes,
        depend_on=["name", "description"],
        thinking=True,
    )

    print(len(seafood_dishes))
    
    # Group by preparation method
    grouped_dishes = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Cluster these seafood dishes by their preparation methods into 'Baked', 'Grilled', 'Fried', 'Raw', 'Sautéed', 'Steamed' and 'Others'.",
        df=seafood_dishes,
        depend_on=["name", "description"],
        thinking=True,
    )
    
    result = (
        grouped_dishes.groupby("cluster_name")
        .agg(dish_count=("name", "count"))
        .reset_index()
    )
    
    prediction = result.values.tolist()
    return prediction


# menu
# select, induce
def pipeline_60():
    query = "Based on menus from dinner services during the 1920s, summarize the characteristics and trends of dinner dishes, including common ingredients, preparation methods, and price ranges."
    answer = """
        Dinner in the 1920s (as reflected in these menus) mixed classic hotel cuisine with regional and international influences, often served in multi-course format (broth/consommé or oyster starter, fish, meat/poultry, sides and salad, dessert, coffee).
        
        Common ingredients:
        - Proteins: chicken (broiled, Maryland, patties), turkey, duck/squab; beef (sirloin, tenderloin, prime rib); pork and mutton/lamb chops; calf’s sweetbreads; abundant seafood—salmon, halibut, sole, striped bass, sand dabs; shellfish (oysters in many styles; crab Louie/cream/au gratin/Creole; lobster Newburg/Thermidor/broiled half).
        - Vegetables and sides: potatoes in many forms (mashed, baked, lyonnaise, sauté, Parisienne, new in cream); peas, asparagus (often with Hollandaise), corn (Maine corn/creamed), spinach with egg, cucumbers, lettuce and tomato salads; artichoke hearts.
        - Sauces/dressings: rich European sauces—Hollandaise, Maitre d’Hotel butter, sauce suprême, Colbert, Financière; Newburg, Thermidor, Cardinal; au gratin; Spanish- and curry-style preparations; mayonnaise and Thousand Island for salads/cocktails; French dressing; gravy.
        - Relishes and starters: celery, olives, gherkins, chow-chow/India relish; oyster cocktails and pan roasts; consomme and cream soups.
        - Desserts: ice creams, sundaes, parfaits, pies (including Baked Alaska Soufflé), fruit with cream; petits fours.
        
        Preparation methods and service:
        - Dry-heat: broiled, grilled, fried, sauté (incl. sauté sec), roasted, baked; breaded cutlets; “planked” steaks with garniture.
        - Moist/cream-based: cream stews/chowders, creamed seafood/chicken, braises.
        - Cold service: seafood cocktails and mayonnaise-dressed salads (Crab/Lobster/Shrimp Louie), half-cracked crab on ice.
        - Continental flourishes: dishes “under glass,” multi-course set dinners (e.g., “Continental Dinner”).
        
        Notable trends:
        - Strong seafood and shellfish presence, especially West Coast styles (sand dabs; Hangtown Fry; Crab/Lobster Louie) alongside French hotel classics (Newburg, Thermidor, Hollandaise) and luxury sweetbreads.
        - Visible Central/Eastern European influence: Chicken Paprikash; Paprika Schnitzel; “Hongroise.”
        - Cream- and butter-rich sides and sauces; artichoke hearts and asparagus as premium garnishes.
        - Prohibition-era drink lists emphasize nonalcoholic choices (ginger ales, sarsaparilla, Coca‑Cola, fruit cups/juices, malted milk), though some menus still list beer.
        
        Typical price ranges (USD):
        - Starters/soups: $0.20–$0.40 per cup; oyster plates/cocktails roughly $0.40–$0.85 (e.g., fried/broiled half‑dozen $0.60–$0.85).
        - Salads/cold seafood: $0.35–$0.75 (Crab Louie ~ $0.60; Lobster Louie ~$0.75).
        - Fish/seafood entrées: about $0.50–$1.25 (sand dabs $0.50–$0.65; salmon $0.60–$0.70; lobster Newburg/Thermidor $1.00–$1.25; half lobster broiled ~$0.85).
        - Meat/poultry entrées: about $0.70–$2.25 (pork or lamb chops ~$0.70; mutton chops ~$0.75; chicken Maryland ~$1.10; steak $1.10–$1.25; whole broiled chicken up to $2.25; sweetbreads ~$0.80–$1.10).
        - Vegetables/sides: $0.15–$0.30 (potatoes $0.15–$0.30; asparagus ~$0.30–$0.40).
        - Desserts: $0.20–$0.50 (specialties like Baked Alaska ~$0.75).
        - Coffee/tea and soft drinks: coffee/tea $0.10–$0.25; bottled sodas/ginger ales ~$0.35–$0.60; elaborate fruit “cups” $1.25–$2.00; milk ~$0.15.
        
        In short, 1920s dinner menus balanced luxe French technique and rich sauces with a marked seafood focus and growing international (notably Hungarian) touches, served with creamy, potato‑heavy sides and fresh lettuce salads—at prices usually under $1.25 for most entrées, with premium poultry/steak and lobster reaching the top of the range.
    """
    
    menu = pd.read_csv("./databases/menu/Menu.csv")
    dish = pd.read_csv("./databases/menu/Dish.csv")
    menu_item = pd.read_csv("./databases/menu/MenuItem.csv")
    menu_page = pd.read_csv("./databases/menu/MenuPage.csv")
    
    # Filter 1920s menus
    menu["date"] = pd.to_datetime(menu["date"],errors="coerce")
    menus_1920s = menu[
        (menu["date"].dt.year >= 1920) & 
        (menu["date"].dt.year <= 1929)
    ]
    menus_1920s = menus_1920s.dropna(subset=['event'])
    print(menus_1920s[['event']])
    
    # Filter breakfast menus
    breakfast_menus = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This menu is for dinner service.",
        df=menus_1920s,
        depend_on=["event"],
        thinking=True,
    )
    print(len(breakfast_menus))

    breakfast_menus = breakfast_menus.rename(columns = {"id":"menu_id"})
    menu_page = menu_page.rename(columns = {"id":"menu_page_id"})
    dish = dish.rename(columns = {"id":"dish_id", "name": "dish_name"})

    merged_df = pd.merge(breakfast_menus, menu_page, on = "menu_id")
    merged_df = pd.merge(merged_df, menu_item, on="menu_page_id")
    merged_df = pd.merge(merged_df, dish, on="dish_id")

    print(len(merged_df))
    
    # Get unique breakfast dishes
    dinner_dishes = merged_df[["dish_name", "description", "price", "first_appeared"]].drop_duplicates()
    print(dinner_dishes.shape[0])
    print(dinner_dishes)
    
    # Summarize characteristics
    prediction = LogicalInduce(operand_type=OperandType.ROW).execute(
        ImplType.LLM_ALL,
        condition="Summarize the characteristics and trends of dinner dishes from the 1920s, including common ingredients, preparation methods, and price ranges.",
        df=dinner_dishes,
        thinking=True,
    )
    
    return prediction

# olympics
# select + cluster
def pipeline_61():
    query = "Filter Asian cities that have held the Olympic Games, group them by country, and count how many times each country has hosted the Olympic Games."
    answer = [["China", 1], ["Japan", 3], ["South Korea", 1]] 

    city = pd.read_csv("./databases/olympics/city.csv")
    game = pd.read_csv("./databases/olympics/games.csv")
    games_city = pd.read_csv("./databases/olympics/games_city.csv")

    south_america_hosts = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This host city is located in Asia.",
        df=city,
        depend_on=["city_name"],
        thinking=True,
    )

    merged_df = pd.merge(south_america_hosts, games_city, left_on='id', right_on='city_id')
    merged_df = pd.merge(merged_df, game, left_on='games_id', right_on='id')

    grouped_hosts = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Group these Asian host cities by their country.",
        df=merged_df,
        depend_on=["city_name"],
        thinking=True,
    )
    print(grouped_hosts.columns)
    result = (
        grouped_hosts.groupby("cluster_name")
        .agg(country_count=("games_name", "count"))
        .reset_index()
    )

    prediction = result.values.tolist()
    return prediction


# olympics
# select
def pipeline_62():
    query = "Which was the earliest Olympic Games that involved any sport originating from the UK?"
    answer = ['1896 Summer', 1896] 

    sports = pd.read_csv("./databases/olympics/sport.csv")
    games = pd.read_csv("./databases/olympics/games.csv")
    events = pd.read_csv("./databases/olympics/event.csv")
    comp_event = pd.read_csv("./databases/olympics/competitor_event.csv")
    games_comp = pd.read_csv("./databases/olympics/games_competitor.csv")

    uk_origin_sports = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This sport originates from the United Kingdom (England, Scotland, Wales, or Northern Ireland).",
        df=sports,
        depend_on=["sport_name"],
        thinking=True,
    )

    m1 = pd.merge(uk_origin_sports.rename(columns={"id": "sport_id"}), events, on="sport_id")
    m2 = pd.merge(m1.rename(columns={"id": "event_id"}), comp_event, on="event_id")
    m3 = pd.merge(m2, games_comp.rename(columns={"id": "competitor_id"}), on="competitor_id")
    m4 = pd.merge(m3, games.rename(columns={"id": "games_id"}), on="games_id")

    earliest = m4.sort_values(by=["games_year", "games_name"], ascending=[True, True]).head(1)
    prediction = earliest[["games_name", "games_year"]].values.tolist()[0]
    return prediction


# formula_1 
# select + select + order
def pipeline_63():
    query = "List all drivers who were born in Europe, participated in the 2017 Monaco Grand Prix, were born after January 1, 1991, and are younger than Alex Albon. Rank them by their debut year, from earliest to latest."
    answer = [['Max', 'Verstappen'], ['Esteban', 'Ocon']]

    drivers = pd.read_csv("./databases/formula_1/drivers.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    results = pd.read_csv("./databases/formula_1/results.csv")

    races  =races[(races['year'] == 2017) & (races['name'] == 'Monaco Grand Prix')]

    drivers['dob'] = pd.to_datetime(drivers['dob'])
    drivers = drivers[drivers['dob'] > '1991-01-01']   
    drivers['dob'] = drivers['dob'].dt.strftime('%Y-%m-%d')
    print(drivers)

    merged_df = pd.merge(races, results, on = "raceId")
    merged_df = pd.merge(merged_df, drivers, on = "driverId")
    print(merged_df)

    op = LogicalSelect(operand_type=OperandType.ROW)
    merged_df = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "The provided birth date is later than the birth date of Alex Albon.",
                        df = merged_df, 
                        depend_on = ['forename', 'surname', 'dob'],
                        thinking = True)

    print(merged_df)

    op = LogicalSelect(operand_type=OperandType.ROW)
    merged_df = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "The nationality corresponds to a country in Europe.",
                        df = merged_df, 
                        depend_on = ['nationality'],
                        thinking = True)

    print(merged_df)

    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(drivers)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The debut year of this formula 1 driver.",
        depend_on=['forename', 'surname'],
        ascending=True,
        k=k,
        df=merged_df,
        sort_algo="heap",
        thinking = True
    )
    
    print(result)
    prediction = result[['forename', 'surname']].values.tolist()
    return prediction


# california_schools
# select + induce
def pipeline_64():
    query = "Summarize the overall SAT performance of the San Francisco Bay Area, based on the average scores for each county."
    answrer = """
        Across the nine Bay Area counties, average SAT section scores cluster near 500 (old 200–800 scale): about 500 in Reading, 510 in Math, and 496 in Writing (unweighted by county size).
        Top overall county averages (mean of Reading, Math, Writing) are Marin (~535), Santa Clara (~534), and San Mateo (~526), followed by Sonoma (~509). Lower averages appear in Solano (~494), Contra Costa (~489), Alameda (~486), Napa (~480), and San Francisco (~465).
        By section: Reading is highest in Marin (~535) and lowest in San Francisco (~459); Math is highest in Santa Clara (~549) and lowest in Napa (~471); Writing is highest in Marin (~533) and lowest in San Francisco (~452).
        Section ranges across counties are sizable—~76 points in Reading, ~78 in Math, and ~81 in Writing—showing notable cross-county variation. Overall pattern: South Bay and Marin lead; San Francisco and Napa trail.
    """
    
    satscores = pd.read_csv("./databases/california_schools/satscores.csv")

    grouped = satscores.groupby("cname").agg(
        AvgScrRead =("AvgScrRead", "mean"),
        AvgScrMath = ("AvgScrMath", "mean"),
        AvgScrWrite =( "AvgScrWrite", "mean")
    ).reset_index();

    print(grouped)

    op = LogicalSelect(operand_type=OperandType.ROW)
    grouped = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This county is in the San Francisco Bay Area.",
                        df = grouped, 
                        depend_on = ['cname'],
                        thinking = True)

    print(grouped)
    
    prediction = LogicalInduce(operand_type=OperandType.ROW).execute(
        ImplType.LLM_ALL,
        condition="Summarize the average SAT score distribution of schools in San Francisco Bay Area.",
        df=grouped,
        thinking=True,
    )
    
    return prediction

# formula_1
# select + impute
def pipeline_65():
    query = "List drivers who were born after 1980, have won a world championship, and have not competed in either GP2 or Formula 2."
    answer = "[['Fernando', 'Alonso'], ['Sebastian', 'Vettel'], ['Max', 'Verstappen']]"

    drivers = pd.read_csv("./databases/formula_1/drivers.csv").dropna()
    drivers = drivers[pd.to_datetime(drivers['dob']).dt.year > 1980]
    drivers = drivers[['forename', 'surname']]
    print(drivers)
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This driver is a Formula 1 world champion.",
                        df = drivers, 
                        depend_on = ['forename', 'surname'],
                        thinking = True)

    result = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition = "This driver has not been raced in GP2 or Formula 2.",
        df = result, 
        depend_on = ['forename', 'surname'],
        thinking = True
    )
    
    prediction = result.values.tolist();
    return prediction


# electronics
# select + select
def pipeline_66():
    query = "In the Amazon table, find the IDs for Sony laptop chargers that originally cost more than £20, using the peak 2020 exchange rate."
    answer = [[758], [2543], [3790]]
    products = pd.read_csv("./databases/electronics/amazon.csv")
    products = products[products['Brand'] == 'Sony']

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This product is a laptop charger.",
                        df = products, 
                        depend_on = ['Name'],
                        thinking = True)

    result = result.dropna(subset=['Original_Price'])
    result = LogicalSelect(operand_type=OperandType.ROW).execute(impl_type=ImplType.LLM_ONE,
                        condition = "This price in USD is larger than 20 GBP using the peak exchange rate of year 2020.",
                        df = result, 
                        depend_on = ['Original_Price'],
                        thinking = True)
    
    prediction = result[['ID']].values.tolist()
    return prediction

# books
# select + select
def pipeline_67():
    query = "Filter ids of books about natural science published between Jan 2000 and Feb 2000, and written in original British English."
    answer = ['8036', '9538']

    books = pd.read_csv("./databases/books/book.csv")
    language = pd.read_csv("./databases/books/book_language.csv")

    books['publication_date'] = pd.to_datetime(books['publication_date'])
    books = books[(books['publication_date'] >= '2000-01-01') & (books['publication_date'] <= '2000-02-29')]
    books['publication_date'] = books['publication_date'].dt.strftime('%Y-%m-%d')

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This book is about natural science.",
                        df = books, 
                        depend_on = ['title'],
                        thinking = True)
    print(result.columns)
    print(result)

    merged_df = pd.merge(result, language, on="language_id")
    merged_df = LogicalSelect(operand_type=OperandType.ROW).execute(
            impl_type=ImplType.LLM_ONE,
            condition = "This language is the original British English.",
            df = merged_df, 
            depend_on = ['language_name'],
            thinking = True
    )
    print(merged_df)
    prediction = merged_df[['book_id']].values.tolist()
    return prediction


# shipping
# select + select
def pipeline_68():
    query = "Which customers received shipments in 2016 and after the 2016 Rio Summer Olympics from a state that is adjacent to their own home state? Answer with the customer name."
    answer = [['Ziebart'], ["Ballew's Hitch Truck & RV"], ['Scherer Truck Equipment Inc'], ['Trailer & RV Parts Warehouse'], ['Sunguard Window Tinting & Truck Accessories'], ['U-haul Center Of Sw Detroit'], ['Shealy Mack Leasing'], ['Fellhoelter Enterprises'], ['U-Haul CO - Moving Center, Clarksville'], ['Mid Town Rent-A-Car'], ['Ace Trailer & Equipment Co., Inc.'], ['Line-X of Metairie'], ['R V City'], ['NW RV Supply']]
    
    shipment = pd.read_csv("./databases/shipping/shipment.csv")
    city = pd.read_csv("./databases/shipping/city.csv")
    customer = pd.read_csv("./databases/shipping/customer.csv")

    print(len(shipment))

    shipment = shipment[(pd.to_datetime(shipment['ship_date']).dt.year == 2016)]
    shipment = LogicalSelect(operand_type=OperandType.ROW).execute(
                    impl_type=ImplType.LLM_ONE, 
                    condition="The date is after the end of Rio Summer Olympics.", 
                    df = shipment, 
                    depend_on = ['ship_date'],
                    thinking = True
                    )

    print(len(shipment))

    shipment = pd.merge(shipment, city, on='city_id')
    shipment = pd.merge(shipment, customer, on='cust_id')
    print(shipment.columns)

    op = LogicalSelect(operand_type=OperandType.ROW)
    df = op.execute(impl_type=ImplType.LLM_ONE, 
                    condition="The two states are geographically adjacent.", 
                    df = shipment, 
                    depend_on = ['state_x', 'state_y'],
                    thinking = True)

    prediction = df[['cust_name']].drop_duplicates().values.tolist()
    return prediction

# car_retails
# select + impute
def pipeline_69():
    query = "Among orders that have been shipped and specified a preferred delivery company in the comments, how many days did it take to ship these orders on average?"
    answer = 3.3

    orders = pd.read_csv("./databases/car_retails/orders.csv")
    orders = orders[orders["status"] == "Shipped"]
    orders = orders.dropna(subset=["comments"])

    op = LogicalSelect(operand_type=OperandType.ROW)
    orders = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This comment appoints or requests specific company for delivering.",
        df=orders,
        depend_on=["comments"],
    )
    print(orders)

    truth = ((pd.to_datetime(orders['shippedDate']) - pd.to_datetime(orders['orderDate'])).dt.days).tolist()
    print(len(truth))
    op = LogicalImpute(operand_type=OperandType.COLUMN)
    imputed_table = op.execute(impl_type=ImplType.LLM_ONE, 
                               condition="Calculate the number of days between shippedDate and orderDate", 
                               df = orders,
                               depend_on = ['shippedDate', 'orderDate'], 
                               new_col = 'days',
                               thinking = True)
    imputed_table['days'] = imputed_table['days'].map(int)
    pred = imputed_table['days'].tolist()

    avg = sum(pred) / len(pred) if pred else 0
    print(pred)

    return avg
    
# restaurants2
# select + select
def pipeline_70():
    query = "Which Yelp restaurants (votes > 3000) are located in Northeastern cities of the United States and serve cuisines that originate from Asia?"
    answer = [['Joeâs Shanghai'], ['Totto Ramen'], ['Ippudo NY']]
    
    yelp = pd.read_csv("./databases/restaurants2/yelp.csv")
    yelp = yelp[yelp['votes'] > 3000]
    print(yelp.shape[0]) # 22
    yelp = LogicalSelect(operand_type=OperandType.ROW).execute(impl_type=ImplType.LLM_ONE, 
                        condition="The zip belongs to a northeast city of USA", 
                        df = yelp, 
                        depend_on = ['zip'],
                        thinking = True)
    yelp = yelp[['name', 'cuisine']]
    print(yelp)
    
    yelp = LogicalSelect(operand_type=OperandType.ROW).execute(impl_type=ImplType.LLM_ONE, 
                        condition="The cuisine originates from Asia.", 
                        df = yelp, 
                        depend_on = ['cuisine'],
                        thinking = True)

    print(yelp)
    prediction = yelp[['name']].values.tolist()

    return prediction
    
# california_schools
# select + order 
def pipeline_71():
    query = "Filter schools with the top 30 FRPM counts that have a human-sounding name, and rank them geographically from south to north by county. Answer with the school name."
    answer  = pd.DataFrame(['Hector G. Godinez', 'James A. Garfield Senior High', 'John H. Francis Polytechnic'])
    
    schools = pd.read_csv("./databases/california_schools/schools.csv")
    frpm = pd.read_csv("./databases/california_schools/frpm.csv")
    
    merged_df = pd.merge(schools, frpm, on='CDSCode')
    merged_df = merged_df.dropna(subset=['FRPM Count (K-12)'])
    merged_df = merged_df.sort_values(by='FRPM Count (K-12)', ascending=False)[:30]
    merged_df = merged_df[['County', 'School Name']]
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    merged_df = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This schools sounds like a human's name.",
                        df = merged_df, 
                        depend_on = ['School Name'],
                        thinking = True)

    print(merged_df)

    k = len(merged_df)
    result = LogicalOrder(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="the latitude of this county",
        depend_on="County",
        ascending=True,
        k=k,
        df=merged_df,
        sort_algo="heap",
        thinking = True
    )
                      
    prediction = result[['School Name']].values.tolist()
    return prediction


# mondial_geo
# select + order
def pipeline_72():
    query = "List lakes with a altitude larger than 500 and whose areas are larger than the area of Shanghai, and rank them based on their absolute distance to London (United Kingdom) in ascending order. Answer with the lake name."
    answer = "[['Lake Bangweulu'], ['Lake Tanganjika'], ['Lake Titicaca'], ['Lake Victoria'], ['Salar de Uyuni']]"
    
    lake = pd.read_csv("./databases/mondial_geo/lake.csv")
    lake = lake[lake['Altitude'] > 500]
    print(lake.shape) # 43
    op = LogicalSelect(operand_type=OperandType.ROW)
    lake = op.execute(impl_type = ImplType.LLM_ONE, 
                    condition = "The Area is larger than the area of Shanghai",
                    df = lake,
                    depend_on = ['Area'],
                    thinking = True)

    print(lake)

    k = len(lake)
    result = LogicalOrder(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="the absolute distance of this lake to London(United Kingdom).",
        depend_on="Name",
        ascending=True,
        k=k,
        df=lake,
        sort_algo="heap",
        thinking = True
    )

    print(result)

    prediction = result[['Name']].values.tolist()
    return prediction

# disney
# select + select + select
def pipeline_73():
    query = "List Disney movies released in the 21st Century that have a male villain and whose total gross is more than 95% of their inflation-adjusted gross. Answer with the movie name."
    answer = [['Frozen'], ['Big Hero 6']]

    characters = pd.read_csv("./databases/disney/characters.csv")
    director = pd.read_csv("./databases/disney/director.csv")
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")

    print(len(characters))
    characters = LogicalSelect(OperandType.ROW).execute(impl_type=ImplType.LLM_ONE, 
                            condition = "The Disney movie villian is male.",
                            df = characters, 
                            depend_on = ['villian'], 
                            thinking=True)

    print(len(characters))
    print(characters)

    characters = LogicalSelect(OperandType.ROW).execute(impl_type=ImplType.LLM_ONE, 
                            condition = "The movie is released in 21th Century.",
                            df = characters, 
                            depend_on = ['release_date'], 
                            thinking=True)
    
    print(len(characters))
    print(characters)

    merged_df = pd.merge(director, characters, left_on="name", right_on="movie_title")
    merged_df = pd.merge(merged_df, movies_total_gross, on = "movie_title")

    print(merged_df)

    merged_df = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="total_gross is larger than 95% of the inflation_adjusted_gross.",
        df=merged_df,
        depend_on=['total_gross', 'inflation_adjusted_gross'],
        thinking=True,
    )

    print(merged_df)
    prediction = merged_df[['movie_title']].values.tolist()
    return prediction
    

# formula_1
# select + select + cluster
def pipeline_74():
    query = "Group Formula 1 circuits by continent, but only include those built after the Vietnam War and that have hosted races for modern (21st-century) constructors."
    answer = [['Sepang International Circuit', 'Asia'], ['Shanghai International Circuit', 'Asia'], ['Bahrain International Circuit', 'Asia'], ['Circuit de Barcelona-Catalunya', 'Europe'], ['Istanbul Park', 'Europe'], ['Hungaroring', 'Europe'], ['Valencia Street Circuit', 'Europe'], ['Marina Bay Street Circuit', 'Asia'], ['Yas Marina Circuit', 'Asia'], ['Circuit Gilles Villeneuve', 'North America'], ['A1-Ring', 'Europe'], ['Circuito de Jerez', 'Europe'], ['Okayama International Circuit', 'Asia'], ['Adelaide Street Circuit', 'Oceania'], ['Phoenix street circuit', 'North America'], ['Korean International Circuit', 'Asia'], ['Autódromo Internacional Nelson Piquet', 'South America'], ['Detroit Street Circuit', 'North America'], ['Fair Park', 'North America'], ['Long Beach', 'North America'], ['Las Vegas Street Circuit', 'North America'], ['Buddh International Circuit', 'Asia'], ['Circuit of the Americas', 'North America'], ['Sochi Autodrom', 'Europe'], ['Baku City Circuit', 'Asia']]

    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    constructors = pd.read_csv("./databases/formula_1/constructors.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    constructorStandings = pd.read_csv("./databases/formula_1/constructorStandings.csv")


    print(len(constructors))
    constructors = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This constructor have participated in at least one Grand Prix after 21st.",
        df=constructors,
        depend_on=['constructorRef', 'name'],
        thinking=True,
    )

    print(len(constructors))
    print(len(circuits))

    circuits = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This circuit is build after the end year of Vietnam War.",
        df=circuits,
        depend_on=['circuitRef', 'name'],
        thinking=True,
    )

    print(len(circuits))

    circuits = circuits.rename(columns={"name": "circuit_name"})
    constructors = constructors.rename(columns={"name": "constructor_name"})
    merged_df = pd.merge(races, circuits, on="circuitId")
    merged_df = pd.merge(merged_df, constructorStandings, on="raceId")
    merged_df = pd.merge(merged_df, constructors, on="constructorId")

    merged_df = merged_df[['circuit_name', 'country']].drop_duplicates()
    print(merged_df)
    grouped_df = LogicalCluster(operand_type=OperandType.ROW).execute(
                        impl_type=ImplType.LLM_ONE,
                        condition="Cluster the circuit by continent.", 
                        df = merged_df,
                        depend_on = ['circuit_name', 'country'],
                        thinking = True
    )

    print(grouped_df)
    print(grouped_df.columns)

    prediction = grouped_df[['circuit_name', 'cluster_name']].values.tolist()

    return prediction


# student_club
# select + select + impute
def pipeline_75():
    query = "Calculate the proportion of the remaining money to the budget amount for those monthly speaking events held before the day COVID-19 was declared a pandemic by the World Health Organization."
    answer = [[1.0], [1.0], [1.0], [1.0], [0.0], [0.4], [0.3], [0.2]]

    budget = pd.read_csv("./databases/student_club/budget.csv")
    event = pd.read_csv("./databases/student_club/event.csv")

    print(len(event))
    event = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This event happens before the declare day of COVID-19 by the World Health Organization.",
        df=event,
        depend_on=['event_date'],
        thinking=True,
    )

    print(event)
    event = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This event is a monthly speaking.",
        df=event,
        depend_on=['event_name'],
        thinking=True,
    )
    print(len(event))
    print(event)
    merged_df = pd.merge(event, budget, left_on="event_id", right_on="link_to_event")
    merged_df.to_csv('output.csv')
    merged_df = LogicalImpute(OperandType.COLUMN).execute(ImplType.LLM_ONE, 
                condition="impute the proportion of remaining money with respect to the amount.", 
                df = merged_df,
                depend_on = ['spent' ,'remaining', 'amount'], 
                new_col = 'propotion', 
                thinking = True)

    print(len(merged_df))
    print(merged_df)

    merged_df['propotion'] = pd.to_numeric(merged_df['propotion'], errors='coerce').round(1)
    prediction = merged_df[['propotion']].values.tolist()
    return prediction


# mondial_geo
# select + cluster + induce
def pipeline_76():
    query = "Summarize the continental distribution of seas that are deeper than half the height of Mt. Everest."
    answer = """
    By continent, the seas in this dataset are distributed as follows:

    - Africa: Atlantic Ocean (western coasts and islands), Mediterranean Sea (north), Indian Ocean (eastern/southern coasts), Gulf of Aden (northeast via Djibouti and Somalia).
    - Asia: Pacific Ocean (Japan, Indonesia, Philippines, Taiwan), Indian Ocean (India, Maldives, Sri Lanka, etc.), Mediterranean Sea (Turkey and the Levant), Arabian Sea (India, Oman, Pakistan), Gulf of Aden (Yemen), and regional seas—South China Sea, Sulawesi Sea, Sunda Sea.
    - Europe: Atlantic Ocean (Portugal, Spain, France, UK, Iceland, etc.), Arctic Ocean (Russia, Svalbard), Mediterranean Sea (Italy, Spain, Greece, Balkans, etc.), Pacific Ocean (via Russia in this table).
    - North America: Arctic Ocean (U.S., Greenland, Canada), Atlantic Ocean (U.S./Canada and Caribbean territories), Caribbean Sea (widespread across islands and Central America), Pacific Ocean (Canada, U.S., Mexico, and Central America).
    - South America: Atlantic Ocean (Brazil, Argentina, etc.), Pacific Ocean (Peru, Chile, Colombia, Ecuador), Caribbean Sea (Colombia, Venezuela).
    - Oceania: Pacific Ocean (dominant—Australia, New Zealand, and many island states/territories), Indian Ocean (Australia).

    Overall pattern: the Atlantic spans Africa, Europe, and the Americas; the Pacific spans Asia, North and South America, and Oceania (and Russia-as-Europe here); the Indian Ocean links Africa, Asia, and Oceania; the Mediterranean is shared by Europe, Africa, and Asia; the Arctic is present for North America and Europe; the Caribbean Sea is concentrated around North and South America; and several regional seas (South China, Sulawesi, Sunda, Arabian, Gulf of Aden) are concentrated in Asia/Africa as shown.
    """

    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    geo_sea = pd.read_csv("./databases/mondial_geo/geo_sea.csv")
    country = pd.read_csv("./databases/mondial_geo/country.csv")

    sea = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This sea is deeper than half of the height of Mt Everest peak.",
        df=sea,
        depend_on=['Name', 'Depth'],
        thinking=True,
    )

    print(sea)

    sea = sea.rename(columns={"Name": "Sea_Name"})
    country = country.rename(columns={"Name":"Country_Name"})
    merged_df = pd.merge(sea, geo_sea, left_on="Sea_Name", right_on="Sea")
    merged_df = pd.merge(merged_df, country, left_on="Country", right_on="Code")

    merged_df = merged_df[['Sea_Name', 'Country_Name']].drop_duplicates()
    
    print(merged_df)

    grouped_df = LogicalCluster(operand_type=OperandType.ROW).execute(
                        impl_type=ImplType.LLM_ONE,
                        condition="Cluster the country by continent.", 
                        df = merged_df,
                        depend_on = ['Country_Name'],
                        thinking = True
    )

    print(grouped_df)
    
    prediction = LogicalInduce(operand_type=OperandType.ROW).execute(
        ImplType.LLM_ALL,
        condition="Summarize the distribution of those seas around each continent.",
        df=grouped_df,
        thinking = True
    )

    return prediction
   

# books
# select
def pipeline_77():
    query = "How many books are written in the languages originating from an Asian country."
    answer = 63

    book_language = pd.read_csv("./databases/books/book_language.csv")
    book = pd.read_csv("./databases/books/book.csv")

    op = LogicalSelect(operand_type=OperandType.ROW)
    book_language = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This language originates from an Asian country.",
                        df = book_language, 
                        depend_on = ['language_name'],
                        thinking = True)

    print(book_language)
    asian_books = pd.merge(book, book_language, on='language_id')
    print(asian_books)
    prediction = int(asian_books['book_id'].nunique())

    return prediction


# california_schools
# select
def pipeline_78():
    query = "List Schools with Top 20 Enrollment and locates in a county in the North of Los Angeles. Answer with the school name."
    answer = [['Visions In Education'], ['Central High East Campus'], ['James Logan High']]

    schools = pd.read_csv("./databases/california_schools/schools.csv")
    frpm = pd.read_csv("./databases/california_schools/frpm.csv")

    merged_df = pd.merge(schools, frpm, on='CDSCode')
    merged_df = merged_df.dropna(subset=['Enrollment (K-12)'])
    merged_df = merged_df.sort_values(by='Enrollment (K-12)', ascending=False)[:20]
    merged_df = merged_df[['County', 'School Name']]

    print(merged_df)
    
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition = "This County is in the North of Los Angeles.",
                        df = merged_df, 
                        depend_on = ['County'],
                        thinking = True)
    prediction = result[['School Name']].values.tolist()
    return prediction


# restaurants2
# select
def pipeline_79():
    query = "How many zomato restaurants have reviewcount over 50 and are locates in a city in the east coast of the United States?"
    answer = 37
    
    zomato = pd.read_csv("./databases/restaurants2/zomato.csv")
    zomato = zomato[zomato['reviewcount'] > 50]
    print(zomato.shape[0]) # 22
    op = LogicalSelect(operand_type=OperandType.ROW)
    zomato = op.execute(impl_type=ImplType.LLM_ONE, 
                        condition="The zip belongs to a city in the east coast of USA.", 
                        df = zomato, 
                        depend_on = ['zip'],
                        thinking = True)
    prediction = zomato.shape[0]
    return prediction


# beer_factory
# select
def pipeline_80():
    query = "How many customers in Sacramento have an email address that includes their full last name?"
    answer = 192

    customer = pd.read_csv("./databases/beer_factory/customers.csv")
    customer = customer[customer['City'] == 'Sacramento']
    print(len(customer))

    customer = LogicalSelect(operand_type=OperandType.ROW).execute(impl_type=ImplType.LLM_ONE, 
                        condition="The email address contains the full last name of this person.", 
                        df = customer, 
                        depend_on = ['Last', 'Email'],
                        thinking = True)
    customer = customer[['CustomerID']].drop_duplicates()
    prediction = customer.shape[0]
    return prediction


# disney
# match
def pipeline_81():
    query = "For years when Disney's total revenue exceeded 35,000, list the release dates of the movies from those years."
    answer = [['30-Mar-07', 2007], ['21-Nov-08', 2008], ['11-Dec-09', 2009], ['24-Nov-10', 2010], ['15-Jul-11', 2011], ['2-Nov-12', 2012], ['27-Nov-13', 2013], ['7-Nov-14', 2014], ['4-Mar-16', 2016], ['23-Nov-16', 2016]]
    characters = pd.read_csv("./databases/disney/characters.csv")
    revenue = pd.read_csv("./databases/disney/revenue.csv")
    revenue = revenue[revenue['Total'] > 35000]
    print(len(characters), len(revenue))
    

    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_SEMI,
                            condition="The 'release_date' corresponds to the 'Year'", 
                            left_df=characters, 
                            right_df=revenue, 
                            left_on='release_date', 
                            right_on='Year',
                            thinking = True) 
    prediction = merged_df[['left_release_date', 'right_Year']].values.tolist()
    return prediction


# codebase_community
# match 
def pipeline_82():
    query = "List the IDs of comments with a score above 40 that discuss any of the top 10 tags with the most count number."
    answer = [[1915], [167835]]
    comments = pd.read_csv("./databases/codebase_community/comments.csv")
    comments = comments[comments['Score'] > 40]
    comments = comments[['Id', 'Text', 'Score']]
    print(len(comments))
    tags =  pd.read_csv("./databases/codebase_community/tags.csv")
    tags = tags.sort_values(by='Count', ascending=False)[:10]

    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_SEMI,
                            condition="The content of the comment 'Text' is related to the 'TagName'", 
                            left_df=comments, 
                            right_df=tags, 
                            left_on='Text', 
                            right_on='TagName',
                            thinking = True) 
    prediction = merged_df[['left_Id']].values.tolist()
    return prediction


# shipping
# match
def pipeline_83():
    query = "For the top 5 customers by annual revenue, which ones are in a state that borders a state containing a city with over 600,000 people? List the customer ID, their home state, and the neighboring state."
    answer = [[954, 'LA', 'Texas'], [1724, 'OH', 'Pennsylvania'], [2421, 'WI', 'Illinois'], [4426, 'LA', 'Texas']]
    customer = pd.read_csv("./databases/shipping/customer.csv")
    customer = customer.sort_values(by='annual_revenue', ascending=False)[:5]
    customer = customer[['cust_id', 'cust_name', 'state']]

    city = pd.read_csv("./databases/shipping/city.csv")
    city = city[city['population']>600000]
    city = city[['state']].drop_duplicates()
    print(len(customer))
    print(len(city))
    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(impl_type=ImplType.LLM_SEMI,
                            condition="The 'left_state' is adjacent to the 'right_state'", 
                            left_df=customer, 
                            right_df=city, 
                            left_on='state', 
                            right_on='state',
                            thinking = True) 
    prediction = merged_df[['left_cust_id', 'left_state', 'right_state']].values.tolist()
    return prediction


# student_club
# impute
def pipeline_84():
    query = "For every meeting with open status, what season does it take place in?"
    answer = ['winter', 'winter', 'winter', 'winter', 'autumn', 'spring']
    event = pd.read_csv("./databases/student_club/event.csv")
    event = event[(event['status'] == 'Open') & (event['type'] == 'Meeting')]
    op = LogicalImpute(OperandType.COLUMN)
    event = op.execute(impl_type=ImplType.LLM_ONE, 
                condition="impute the season of the event", 
                df = event,
                depend_on = ['event_name', 'event_date'], 
                new_col = 'season', 
                thinking = True)
    prediction = event['season'].tolist()
    return prediction


# debit_card_specializing
# impute
def pipeline_85():
    query = "What is the per-item cost for transactions with a price over 2000?"
    answer = ['21.83', '22.55', '23.59', '21.96', '22.157', '23.065', '23.0335', '25.11', '22.73', '23.06', '22.51', '22.56', '23.05', '23.38', '22.64', '22.80']
    transactions = pd.read_csv("./databases/debit_card_specializing/transactions_1k.csv")
    transactions = transactions[transactions['Price'] > 2000]
    print(transactions)
    op = LogicalImpute(OperandType.COLUMN)
    transactions =  op.execute(impl_type=ImplType.LLM_ONE, 
                condition="Impute the unit price", 
                df = transactions,
                depend_on = ['Amount', 'Price'], 
                new_col = 'unitPrice', 
                thinking = True)
    prediction = transactions['unitPrice'].tolist()
    return prediction


# movies
# cluster
def pipeline_86():
    query = "Among the top 10 highest-rated movies in imdb table, count how many of these films were directed by someone from each continent. What are the numbers for each continent?"
    answer = [['Europe Directors', 3], ['North America Directors', 6], ['Oceania Directors', 1]]
    movies = pd.read_csv("./databases/movies/imdb.csv")
    movies = movies.sort_values(by="Rating", ascending=False)[:10]
    movies = movies[['Title', 'Director']]
    print(movies.shape[0])

    op = LogicalCluster(operand_type=OperandType.ROW)
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition="Group by the continent of the director's nationality", 
                        df = movies,
                        depend_on = 'Director',
                        thinking = True)
    print(result)
    prediction = result.groupby("cluster_name").agg(movieCount=("Title", "count")).reset_index().values.tolist()

    return prediction
   

# car_retails
# cluster
def pipeline_87():
    query = "Among the 10 most expensive products with over 100 units in stock, count how many have a model year that is a leap year and how many have a model year that is a common year."
    answer = [['Common Year Cluster', 8], ['Leap Year Cluster', 2]]
    products = pd.read_csv("./databases/car_retails/products.csv")
    products = products[products['quantityInStock'] > 100]
    products = products.sort_values(by="buyPrice", ascending=False)[:10]
    print(products.shape[0])
    op = LogicalCluster(operand_type=OperandType.ROW)
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition="Cluster the year of the product into 'Leap Year' or 'Common Year'.", 
                        df = products,
                        depend_on = 'productName',
                        thinking = True)
    prediction = result.groupby("cluster_name").agg(productCount=("productCode", "count")).reset_index().values.tolist()              
    return prediction


# european_football_2
# order
def pipeline_88():
    query = "List the 5 tallest football players, and order them by the population size of their home countries, from largest to smallest"
    answer = ['Lacina Traore', 'Kristof van Hout', 'Nikola Zigic', 'Vanja Milinkovic-Savic', 'Bogdan Milic']
    player = pd.read_csv("./databases/european_football_2/Player.csv")
    player = player.sort_values(by="height", ascending=False)[:5]
    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(player)
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition="the population of the home country of the football player listed in 'player_name' column.",
                        depend_on= 'player_name',
                        ascending = False,
                        k = k,
                        df=player, 
                        sort_algo='simple',
                        thinking = True)
    print(result)
    predication = result['player_name'].tolist()
    return predication
   

# mondial_geo
# order
def pipeline_89():
    query = "Among the Top 10 most populous countries, list those have an area greater than 1 million square kilometers, and rank them based on the location of their capital from the northest to the southest."
    answer = ['Russia', 'China', 'United States', 'India', 'Indonesia', 'Brazil']
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    country = country.sort_values(by="Population", ascending=False)[:10]
    print(country)
    country = country[country['Area'] > 1000000]
    print(country)

    op = LogicalOrder(operand_type=OperandType.ROW)
    k = len(country)
    result = op.execute(impl_type=ImplType.LLM_ONE,
                        condition="the latitude of capital of this country.",
                        depend_on= 'Name',
                        ascending = False,
                        k = k,
                        df=country, 
                        sort_algo='simple',
                        thinking = True)
    print(result)
    prediction = result['Name'].tolist()
    return prediction
    

# authors
# select
def pipeline_90():
    query = "List the IDs of authors who are from the University of Illinois Chicago and have Chinese-style names."
    answer = [[93570], [220256]]
    author = pd.read_csv("./databases/authors/Author.csv")
    author = author[author['Affiliation'] == 'University of Illinois Chicago']
    print(len(author))
    print(author)

    author = LogicalSelect(operand_type=OperandType.ROW).execute(impl_type=ImplType.LLM_ONE, 
                        condition="The is a Chinese style name.", 
                        df = author, 
                        depend_on = ['Name'],
                        thinking = True)
    print(author)
    prediction = author[['Id']].values.tolist()
    return prediction


# authors
# select 
def pipeline_91():
    query = "List conferences whose full names start with the prefix 'ACM' and that are about data engineering. Answer with the conference full name."
    answer = [['ACM International Workshop on Data Engineering for Wireless and Mobile Access']]
    conference = pd.read_csv("./databases/authors/Conference.csv")
    conference = conference.dropna(subset= ['FullName'])
    conference = conference[conference['FullName'].str.startswith('ACM')]
    print(len(conference))
    print(conference['FullName'])

    conference = LogicalSelect(operand_type=OperandType.ROW).execute(impl_type=ImplType.LLM_ONE, 
                        condition="The is a conference about data enginnering.", 
                        df = conference, 
                        depend_on = ['FullName'],
                        thinking = True)

    print(conference['FullName'])
    prediction = conference[['FullName']].values.tolist()
    return prediction


# authors
# impute
def pipeline_92():
    query = "Infer the research domain of the conference 'International Conference on Data Engineering' based on the name."
    answer = "-"
    conference = pd.read_csv("./databases/authors/Conference.csv")
    conference = conference[conference['FullName'] == 'International Conference on Data Engineering']
    print(len(conference))
    print(conference['FullName'])

    result = LogicalImpute(operand_type=OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Impute the research domain of this ACM conference in computer science.",
        df=conference,
        depend_on="FullName",
        new_col="domain",
        thinking = True
    )

    prediction = result[['FullName', 'domain']].values.tolist()
    return prediction

# european_football_2
# order
def pipeline_93():
    query = "List leagues whose countries have Top 5 largest land area."
    answer = ['France Ligue 1', 'Spain LIGA BBVA', 'Germany 1. Bundesliga', 'Poland Ekstraklasa', 'Italy Serie A']
    country = pd.read_csv("./databases/european_football_2/Country.csv")
    leagues = pd.read_csv("./databases/european_football_2/League.csv")

    merged = pd.merge(
        leagues,
        country.rename(columns={"name": "country_name"}),
        left_on="country_id",
        right_on="id",
    )
    print(merged)
    op = LogicalOrder(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="the land area of the country",
        depend_on="country_name",
        ascending=False,
        k=5,
        df=merged,
        sort_algo="simple",
        thinking=True,
    )
    return result["name"].tolist()


# european_football_2
# select
def pipeline_94():
    query = "List the Dutch players among the top 10 tallest players."
    answer = ['Kevin Vink', 'Jurgen Wevers']

    players = pd.read_csv("./databases/european_football_2/Player.csv")
    players = players.sort_values(by="height", ascending=False)[:10]

    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This player is a Dutch.",
        df=players,
        depend_on=["player_name"],
        thinking=True,
    )
    return result["player_name"].tolist()


# european_football_2
# impute
def pipeline_95():
    query = "Among the players born in 1997, list the top 5 players with the highest bmi."
    answer = [('Joris Gnagnon',), ('Pantelis Hatzidiakos',), ('Breel Embolo',), ('Mikel Oiarzabal',), ('Michal Bartowiak',)]

    player = pd.read_csv("./databases/european_football_2/Player.csv")
    player = player[pd.to_datetime(player["birthday"]).dt.year == 1997]
    # print(player.shape[0])
    player = LogicalImpute(OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Calculate the BMI of each player based on the 'height' in cm and the 'weight' in pounds",
        df=player,
        depend_on=["height", "weight"],
        new_col="bmi",
        thinking=True,
    )
    player = player.sort_values("bmi")[:5]
    return player["player_name"].tolist()


# european_football_2
# select + select + induce
def pipeline_96():
    query = "Summarize the attributes of players born in 1997 who are taller and heavier than Kylian Mbappé."
    answer = """
        Scope and data: Young players (mostly born in 1997) with multiple snapshots (2014–2016). Attributes cover technical (passing, shooting, dribbling), defending, physical (pace, strength), mentality, GK stats, work rates, preferred foot, height/weight, and overall/potential ratings.

        Overall and potential:
        - Overall range: ~50–76. Top overall: Breel Embolo 76; then Adalberto Peñaranda 72, Jorge Meré 72, Issa Diop 71, Bartłomiej Drągowski 71, Chidozie Awaziem 70, Donny van de Beek 69, Joe Gomez 68, Cameron Borthwick‑Jackson 67.
        - Potential range: ~66–88. Highest: Embolo 88; then Drągowski 86, Issa Diop 86, Jorge Meré 84, Peñaranda 84, Donny van de Beek 83, Awaziem 83, Joe Gomez 81. Many others sit mid‑70s to ~78 (e.g., Tousart 78, Kownacki 78, Sadiq 77, Tetteh 76, Henrichs 74–75, Bernardoni 77).

        Standouts by role:
        - Attackers/forwards
            • Breel Embolo (76 OVR/88 POT): Elite pace (84 accel/87 sprint), strong (85), very good dribbling (77), finishing 76, high work rate. Physical, direct striker/wing forward profile.
            • Adalberto Peñaranda (72/84): Fast (82/81), very good finisher (78) with shot power 73, dribbling/ball control ~74/73; right‑footer. Mobile attacker.
            • Dawid Kownacki (63/78): Balanced forward; pace 74/75, finishing 67, penalties 69, decent link play.
            • Umar Sadiq (63/77): Tall (193 cm), aerial/target traits (heading 62), finishing 66, decent pace for size (68/68).
            • Benjamin Tetteh (62/76): Tall (193 cm) with strength (81) and pace (79/77); dribbling 65, finishing 62.
        - Midfielders
            • Donny van de Beek (69/83): All‑round CM/AM; short passing/vision 73, ball control 75, shot power 70; balanced physically.
            • Rolando Mandragora (66/76): Left‑footed CM/DM; short/long passing 72/70, ball control 72, good agility/balance.
            • Lucas Tousart (62/78): Defensive CM (high defensive work); strength 82, stamina 72, tackling ~68, solid short passing 67.
            • Luca Checchin (62/74–75): Deep‑lying passer; long/short passing 69/69, set‑piece curve 64, balanced physique.
            • Benjamin Henrichs (58/72–74): Versatile wide/CM skill set; dribbling 66, crossing 58, short passing 59–63.
        - Defenders
            • Issa Diop (71/83–86): Dominant CB; strength 86, jumping 85, tackling ~70–74, heading 73.
            • Jorge Meré (72/84): Smart CB; marking 78, tackling 73–74, interceptions 73; good acceleration/agility for CB.
            • Joe Gomez (68/80–81): Athletic CB/FB; sprint 80, strength 76, tackling ~71–72, crossing/dribbling serviceable.
            • Chidozie Awaziem (70/83): Quick, strong CB; tackling ~71–72, interceptions 73, pace 67/73.
            • Cameron Borthwick‑Jackson (67/79): Left‑back; excellent crossing 73, curve 74, standing/sliding tackle 74/69.
            • Giuseppe Pezzella (60/73): Left‑back; dribbling 65, crossing 63, solid defensive stats around low‑60s.
            • Others: Pantelis Hatzidiakos (61/74) CB with long‑passing/shot power; Mateusz Wieteska (56/68) CB; Taylor Moore (59/75) CB; Joris Gnagnon (59/73–75) strong CB.
        - Goalkeepers
            • Bartłomiej Drągowski (71/86): Top prospect GK; reflexes 76, diving/handling ~68, positioning 71–72.
            • Paul Bernardoni (65/77–79): Reflexes 69, diving 68, handling 65; left‑footed.
            • Vanja Milinković‑Savić (61–62/71–74): Very tall (203 cm); diving 68, reflexes 66.
            • Fabrizio Alastra (56/68–78): Reflexes 63, decent kicking 56.

        Attribute leaders:
        - Pace: Embolo (87 sprint), Peñaranda (81), Gomez (80), Ajeti (76), Kownacki (75), Tetteh (77), Awaziem (73).
        - Dribbling/ball control: Embolo (77/82), Peñaranda (74/73), van de Beek (69/75), Tetteh (65/67), Pezzella (65/67).
        - Finishing/shooting: Peñaranda 78, Embolo 76, Kownacki 67, Sadiq 66; strongest shot power among attackers: Embolo 75.
        - Passing/vision (non‑GK): van de Beek (short 73, vision 73), Mandragora (72/68, long 70), Checchin (short/long 69/69), Tousart (67 short, 62 long).
        - Defending: Meré (marking 78; stand/sliding 73/74), Diop (tackling ~70–74), Borthwick‑Jackson (stand/sliding 74/69), Gomez (71/72), Awaziem (~71/68).
        - Physical strength: Diop 86, Embolo 85, Gomez 76, Tousart 82, Tetteh 81.
        - Aerial: Diop jumping 85; Embolo/Peñaranda heading mid‑60s/70s; tall strikers (Sadiq, Tetteh) useful in air.

        Work rates and footedness: Mostly medium/medium work rate; some high attacking (Embolo, Sadiq, Tetteh). Predominantly right‑footed; notable left‑footers include Borthwick‑Jackson (LB), Mandragora (CM/DM), Bernardoni (GK), Pezzella (LB).
        Physical profiles: Heights cluster ~180–190 cm. Tallest: Vanja Milinković‑Savić (203 cm), Issa Diop/Benjamin Tetteh/Umar Sadiq (193 cm). Lightest: many midfielders/FBs around 168–176 lb (76–80 kg equivalents vary by unit).
        Development trend (2014→2016): Clear growth for top prospects: Embolo (~64→76), Issa Diop (~58→71), Donny van de Beek (~64→69), Jorge Meré (~59→72), Drągowski (~50→71). Forwards like Peñaranda/Kownacki/Sadiq and defenders like Awaziem/Gomez/Borthwick‑Jackson move from low/mid‑60s toward solid first‑team levels.
        Bottom line: This cohort features elite prospects at every line—Embolo (dynamic scorer), Peñaranda (finisher), van de Beek/Mandragora/Tousart (play/build), Meré/Diop/Awaziem/Gomez (defensive spine), and Drągowski (top GK)—with strong athletic profiles (pace/strength) and room to grow toward high‑80s potential in several cases.
    """
    player = pd.read_csv("./databases/european_football_2/Player.csv")
    player_attributes = pd.read_csv(
        "./databases/european_football_2/Player_Attributes.csv"
    )
    print(player[['height', 'weight']])

    player = player[pd.to_datetime(player["birthday"]).dt.year == 1997]
    # player = player[player['height'] > 180]
    # player = player[player['weight'] > 165]
    print(player.shape[0])
    player = LogicalSelect(OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The player's height is taller than Kylian Mbappé's height",
        df=player,
        depend_on="height",
        thinking=True,
    )
    # print(player.shape[0]) 
    player = LogicalSelect(OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The player's weight is heavier than Kylian Mbappé's weight",
        df=player,
        depend_on="weight",
        thinking=True,
    )
    merged = pd.merge(player, player_attributes, on="player_api_id")
    print(merged.shape)  
    print(merged)
    summary = LogicalInduce(OperandType.ROW).execute(
        impl_type=ImplType.LLM_ALL,
        condition="Summarize the attributes of the players.",
        df=merged,
        thinking=True,
    )
    return summary


# electronics
# select + induce
def pipeline_97():
    query = "Summarize the common features of Samsung products released after 2012 from Amazon table."
    answer = """
        Common features of Samsung products released after 2012 are listed as follows:
        - Product scope: Samsung laptops (Chromebook 2, ATIV Book 9) and a broad set of compatible accessories (AC adapters, batteries, shells/sleeves, replacement screens, HDD, peripherals).
        - Displays: LED-lit 11.6–15.6-inch panels are common; HD 1366×768 up to Full HD 1920×1080; some touch models; replacement screens often 1366×768.
        - Performance: CPUs include Intel Celeron, Core i5-4200U, Core M 5Y31, and Samsung Exynos; memory is DDR3L (2–8 GB); storage favors SSDs (16–256 GB), with an optional 2.5-inch SATA III 2 TB HDD.
        - OS & battery life: Chrome OS (Chromebook) and Windows 8.1 (ATIV), with advertised long battery life (about 8–10.5 hours).
        - Power/charging: Most adapters accept 100–240V input and output 19V (≈2.1–4.74A; 40–90W); many emphasize extra-long ~14 ft cords, safety protections (over-voltage/short-circuit/overheating), and 24‑month warranties.
        - Batteries: Li‑ion packs, typically 11.1V (10.8V compatible) with 4400–5200 mAh capacities; 1–2 year warranties common; broad compatibility with Samsung Series/NP models.
        - Protection/carry: Hard polycarbonate snap-on cases for Samsung Chromebook 2 (XE503C12), lightweight sleeves, and messenger bags; designs stress shock/ scratch protection and ventilation.
        - Peripherals: Bluetooth laser projection keyboard and a 23,000 mAh solar power bank with multi-voltage outputs (5/12/16/19V).
        - Design themes: Lightweight ultrabook builds, no optical drives, LED-lit screens, Intel HD graphics on Windows models, and black/silver finishes.
    """
    amazon = pd.read_csv("./databases/electronics/amazon.csv")
    amazon = amazon[amazon["Brand"] == "Samsung"]
    amazon = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The product in released after 2012",
        df=amazon,
        depend_on="Name",
        thinking=True,
    )
    print(amazon.shape[0])
    print(amazon)
    
    summary = LogicalInduce(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ALL,
        condition="Summarize the common features of the products",
        df=amazon,
        thinking=True,
    )

    return summary


# disney
# select + impute + induce
def pipeline_98():
    query = "Among all movies directed by Wolfgang Reitherman released before Titanic, describe the main plots of the movie with the highest annual inflation-adjusted gross rate relative to its release date and 2025."
    answer = """
        In the Hundred Acre Wood, Winnie the Pooh’s love of honey leads him to overeat at Rabbit’s house and get stuck in the doorway, setting a playful tone for storybook adventures narrated from the pages. 
        A blustery day brings gusts that send Piglet flying, topple Owl’s house, and lead to a flood where Pooh and friends rescue Piglet; Pooh is celebrated as a hero after a surreal dream of Heffalumps and Woozles. 
        Tigger arrives with boundless bouncing that causes mischief, culminating in Tigger and Roo getting stuck in a tree and a failed plan to curb Tigger’s bouncing; in the end, his friends accept him as he is. 
        The film stitches together these episodes with warm narration and closes on a gentle note as Christopher Robin and Pooh reflect on growing up and the enduring nature of their friendship.
    """
    movies_total_gross = pd.read_csv("./databases/disney/movies_total_gross.csv")
    director = pd.read_csv("./databases/disney/director.csv")
    # movies_total_gross = movies_total_gross[movies_total_gross['genre'] == 'Drama']
    director = director[director["director"] == "Wolfgang Reitherman"]
    merged = pd.merge(
        left=movies_total_gross, right=director, left_on="movie_title", right_on="name"
    )
    # print(merged.shape[0]) # 9

    merged = LogicalSelect(OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The movie is released before Titanic",
        df=merged,
        depend_on="release_date",
    )
    print(merged)  # 3
    merged = LogicalImpute(OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Calculate the annual inflation rate in decimal in gross relative to its release date and 2025. Answer ONLY with a float number within 0 and 1.",
        df=merged,
        depend_on=["total_gross", "inflation_adjusted_gross", "release_date"],
        new_col="annual_inflation_rate",
        thinking=True,
    )
    # To avoid sorting errors due to potential mixed types in 'annual_inflation_rate', coerce to numeric and handle errors
    merged["annual_inflation_rate"] = pd.to_numeric(merged["annual_inflation_rate"], errors="coerce")
    # merged = merged.dropna(subset=["annual_inflation_rate"])
    # merged["annual_inflation_rate"] = merged["annual_inflation_rate"].astype(float)
        
    merged = merged.sort_values("annual_inflation_rate", ascending=False)[:1]
    merged = merged[['movie_title']]
    print(merged)
    description = LogicalInduce(OperandType.ROW).execute(
        impl_type=ImplType.LLM_ALL,
        df=merged,
        condition="Describe the main movie plot of this movie",
        thinking=True,
    )
    return description


# disney
# impute + match
def pipeline_99():
    query = "List the characters released in the year with highest ratio of Studio Entertainment revenue to total revenue."
    answer = ["Pocahontas"]
    characters = pd.read_csv("./databases/disney/characters.csv")
    revenue = pd.read_csv("./databases/disney/revenue.csv")
    revenue = revenue.dropna(subset=["Studio Entertainment[NI 1]", "Total"])
    revenue = LogicalImpute(OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Calculate the ratio of Studio Entertainment revenue to total revenue",
        df=revenue,
        depend_on=["Studio Entertainment[NI 1]", "Total"],
        new_col="ratio",
        thinking=True,
    ) 
    revenue = revenue.sort_values("ratio", ascending=False)[:1]

    print(revenue)  # 1995
    merged = LogicalMatch(OperandType.CELL).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The release date corresponds to the year",
        left_df=characters,
        right_df=revenue,
        left_on="release_date",
        right_on="Year",
    )
    print(merged)
    return merged["left_hero"].tolist()



# nextiaJD
# select
def pipeline_100():
    query = "Among the animals in lost and found created in 2020, how many of them have three colors of fur?"
    answer = 12
    animals = pd.read_csv(
        "./databases/nextiaJD/animal-control-inventory-lost-and-found.csv"
    )
    animals = animals[pd.to_datetime(animals["DateCreated"]).dt.year == 2020]
    print(animals.shape)  # 265
    op = LogicalSelect(operand_type=OperandType.ROW)
    result = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The animal has three colors of fur",
        df=animals,
        depend_on="Color",
    )
    return result.shape[0]


# nextiaJD
# select + match
def pipeline_101():
    query = "How many Theatre/Performance cultural spaces with over 200 seats are located within a 20-minute walking distance of a nearby community center, where both the cultural spaces and community centers are situated at a latitude higher than Vancouver City Hall?"
    answer = 17
    cultural_spaces = pd.read_csv("./databases/nextiaJD/cultural-spaces.csv")
    cultural_spaces = cultural_spaces[cultural_spaces["TYPE"] == "Theatre/Performance"]
    cultural_spaces = cultural_spaces[cultural_spaces["NUMBER_OF_SEATS"] > 200]
    op = LogicalSelect(operand_type=OperandType.ROW)
    cultural_spaces = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This location has a higher latitude than Vancouver City Hall",
        df=cultural_spaces,
        depend_on="Geom",
        thinking=True,
    )
    print(cultural_spaces["ADDRESS"])  # 63

    community_centres = pd.read_csv("./databases/nextiaJD/community-centres.csv")
    op = LogicalSelect(operand_type=OperandType.ROW)
    community_centres = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="This location has a higher latitude than Vancouver City Hall",
        df=community_centres,
        depend_on="Geom",
        thinking=True,
    )
    print(community_centres["ADDRESS"])  # 6

    op = LogicalMatch(operand_type=OperandType.CELL)
    merged_df = op.execute(
        impl_type=ImplType.LLM_SEMI,
        condition="The two addresses are within a 20-minute walk",
        left_df=cultural_spaces,
        right_df=community_centres,
        left_on="ADDRESS",
        right_on="ADDRESS",
        thinking=True,
    )
    merged_df = merged_df[['left_ADDRESS']].drop_duplicates()
    # print(merged_df[['left_ADDRESS', 'right_ADDRESS']])
    return merged_df.shape[0]


# nextiaJD
# select + select + order
def pipeline_102():
    query = "List the Top 1 most-played Taylor Swift songs about love, from Taylor Swift's full-length studio albums in the iTunes table."
    answer = ['Blank Space']

    music = pd.read_csv("./databases/music/itunes.csv")
    music = music[(music["Artist_Name"] == "Taylor Swift")]
    print(music.shape)  # 194
    op = LogicalSelect(operand_type=OperandType.ROW)
    music = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The album is an official full-length studio album by Taylor Swift.",
        df=music,
        depend_on="Album_Name",
        thinking=True,
    )
    print(music.shape)

    music = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The song is mainly about love.",
        df=music,
        depend_on="Song_Name",
        thinking=True,
    )
    print(music.shape)

    op = LogicalOrder(operand_type=OperandType.ROW)
    music = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The song's total play count",
        df=music,
        depend_on="Song_Name",
        k=1,
        ascending=False,
        sort_algo="heap",
    )
    return music["Song_Name"].tolist()


# music
# select + induce
def pipeline_103():
    query = "Summarize the common characteristics of Taylor Swift songs in the 'amazon_music' table, which are released after the 2012 London Olympics."
    answer = """
        Across the listed Taylor Swift tracks, the common characteristics are:
        - Artist and label: All are by Taylor Swift under Big Machine Records, LLC (consistent label and copyright ownership).
        - Pricing: Nearly all tracks are priced at $1.29, with only a rare exception at $0.99 (Invisible).
        - Genres: Predominantly Pop (especially the 1989 era and its karaoke versions), with a substantial subset classified as Country (notably the Red era) and some entries tagged Country Rock/Rock.
        - Release windows: Concentrated around three periods—2009 (early country-era tracks), 2012 (Red), and 2014–2015 (1989 originals, singles, and karaoke versions). Many share the same exact release dates within each era.
        - Song lengths: Generally radio-friendly, clustering around 3:11–4:07; a few longer ballads run 4:29–5:27.
        - Albums and versions: Heavy representation from 1989 (standard tracks appear multiple times, plus a full set of karaoke versions). Additional presence from Red and Fearless Platinum Edition.
        - Collaborations: A minority feature guest artists (Kendrick Lamar, Gary Lightbody, Ed Sheeran), but most tracks are solo.
        
        In short: Taylor Swift songs spanning 2009–2015 on Big Machine Records, mostly $1.29, with lengths around three to four minutes, dominated by Pop from the 1989 era and complemented by earlier Country material, including duplicate listings and karaoke counterparts.
        """
    music = pd.read_csv("./databases/music/amazon_music.csv")
    music = music[(music["Artist_Name"] == "Taylor Swift")]
    print(music.shape)
    op = LogicalSelect(operand_type=OperandType.ROW)
    music = op.execute(
        impl_type=ImplType.LLM_ONE,
        condition="The song is released after the 2012 London Olympics",
        df=music,
        depend_on="Song_Name",
        thinking=True,
    )
    print(music)
    op = LogicalInduce(operand_type=OperandType.ROW)
    music = op.execute(
        impl_type=ImplType.LLM_ALL,
        condition="Summarize the common characteristics of the songs",
        df=music,
        thinking=True,
    )
    return music


# movies
# select + select
def pipeline_104():
    query = "Among movies with rating more than 7.5 in the 'rotten_tomatoes' table, how many of them are released in the same year as Christopher Nolan's Inception, were directed by female directors?"
    answe = 2
    rotten_tomatoes = pd.read_csv("./databases/movies/rotten_tomatoes.csv")
    rotten_tomatoes["Rating"] = pd.to_numeric(
        rotten_tomatoes["Rating"], errors="coerce"
    )
    rotten_tomatoes = rotten_tomatoes[rotten_tomatoes["Rating"] > 7.5]
    print(rotten_tomatoes.shape)  # 220
    rotten_tomatoes = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The movie is released in the same year as Christopher Nolan's Inception",
        df=rotten_tomatoes,
        depend_on="Year",
        thinking=True,
    )

    rotten_tomatoes = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The director is a female director",
        df=rotten_tomatoes,
        depend_on="Director",
        thinking=True,
    )
    return rotten_tomatoes.shape[0]


# movie_3
# select
def pipeline_105():
    query = "How many movies rated 'PG' have a longer runtime than 'The Silence of the Lambs'?"
    answer = 82
    film = pd.read_csv("./databases/movie_3/film.csv")
    film = film[film["rating"] == "PG"]
    # print(film.shape[0]) # 194
    film = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The movie has a longer runtime than 'The Silence of the Lambs'",
        df=film,
        depend_on=["length"],
        thinking=True,
    )
    return film.shape[0]


# movie_3
# select + cluster
def pipeline_106():
    query = "Consider English Action movies rated NC-17 with a 'Reflection' theme. Group them into 'Comedy', 'Thriller', 'Tragedy' and 'Others'. Count the movies within each group."
    answer = [['Comedy', 2], ['Thriller', 4]]
    film = pd.read_csv("./databases/movie_3/film.csv")
    film_category = pd.read_csv("./databases/movie_3/film_category.csv")
    category = pd.read_csv("./databases/movie_3/category.csv")
    language  = pd.read_csv("./databases/movie_3/language.csv")
    language = language.rename(columns={"name": "language_name", "last_update": "language_last_update"})

    language = language[language['language_name'] == 'English']

    film = film[film["rating"] == "NC-17"]
    print(film.shape[0])  # 194
    
    merged = pd.merge(film, film_category, on="film_id")
    merged = pd.merge(merged, category, on="category_id")
    merged = merged[merged["name"] == "Action"]
    print(merged["description"])  # 12
    print(merged.columns)

    reflection_films = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The movie is described as a Reflection",
        df=merged,
        depend_on=["description"],
        thinking=True,
    )
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 0)

    print(reflection_films[['title', 'description']])
    reflection_films = pd.merge(reflection_films, language, on="language_id")

    result = LogicalCluster(operand_type=OperandType.ROW).execute(impl_type=ImplType.LLM_ONE,
                                condition="Group the films into 'Comedy', 'Thriller', 'Tragedy' and 'Others'",
                                df=reflection_films,
                                depend_on=['title', 'description'],
                                thinking=True)

    result = result.groupby("cluster_name").agg(categoryCount=("film_id", "count")).reset_index()
    
    print(result)
    prediction = result.values.tolist()
    return prediction
 

# movie_3
# select
def pipeline_107():
    query = "How many male actors have starred in the film ACADEMY DINOSAUR?"
    answer = 4
    film = pd.read_csv("./databases/movie_3/film.csv")
    film_actor = pd.read_csv("./databases/movie_3/film_actor.csv")
    actor = pd.read_csv("./databases/movie_3/actor.csv")

    film = film[film["title"] == "ACADEMY DINOSAUR"]
    merged = pd.merge(film, film_actor, on="film_id")
    merged = pd.merge(merged, actor, on="actor_id")
    print(merged.shape[0])  # 10
    result = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The actor is an male actor",
        df=merged,
        depend_on=["first_name", "last_name"],
        thinking=True,
    )
    return result.shape[0]


# mondial_geo
# select + select
def pipeline_108():
    query = "List the top 3 European countries with the highest population growth that have a republican government."
    answer = ["Iceland", "San Marino", "Switzerland"]
    politics = pd.read_csv("./databases/mondial_geo/politics.csv")
    population = pd.read_csv("./databases/mondial_geo/population.csv")
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    politics = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The goverment is a republican government",
        df=politics,
        depend_on=["Government"],
        thinking=False,
    )
    print(politics.shape[0])  # 104, 107
    merged = pd.merge(politics, population, on="Country")
    merged = pd.merge(merged, country, left_on="Country", right_on="Code")
    merged = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The country is a European country",
        df=merged,
        depend_on=["Name"],
        thinking=False,
    )
    print(merged.shape[0])  # 13
    merged = merged.sort_values(by="Population_Growth", ascending=False)[:3]
    return merged["Name"].tolist()


# mondial_geo
# select + impute
def pipeline_109():
    query = "Among European countries with more than 200,000 GDP, which one has the lowest Population Density?"
    answer = "Spain"
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    economy = pd.read_csv("./databases/mondial_geo/economy.csv")
    merged = pd.merge(country, economy, left_on="Code", right_on="Country")
    merged = merged[merged["GDP"] > 200000]
    print(merged.shape[0])

    merged = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The country is a European country",
        df=merged,
        depend_on=["Name"],
        thinking=False,
    )
    print(merged.shape[0])
    result = LogicalImpute(operand_type=OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The population density",
        df=merged,
        depend_on=["Area", "Population"],
        new_col="population_density",
        thinking=True,
    )
    result = result.sort_values(by="population_density", ascending=True)[:1]
    return result["Name"].tolist()


# mondial_geo
# cluster
def pipeline_110():
    query = "Among countries with population growth less than 1.0, group them by their continents, and then list the average GDP of 'Asia', 'Europe' and 'North America' accordingly."
    answer = ["856275", "213834", "411848"]
    country = pd.read_csv("./databases/mondial_geo/country.csv")
    economy = pd.read_csv("./databases/mondial_geo/economy.csv")
    population = pd.read_csv("./databases/mondial_geo/population.csv")
    population = population[population["Population_Growth"] < 1.0]
    print(population.shape[0])
    merged = pd.merge(population, economy, on="Country")
    merged = pd.merge(merged, country, left_on="Country", right_on="Code")

    result = LogicalCluster(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Cluster the countries by their continents into 'Asia', 'Europe', 'America' and 'Others'",
        df=merged,
        depend_on=["Name"],
        thinking = True,
    )
    result = result.groupby("cluster_name")["GDP"].mean()
    return result[["Asia", "Europe", "America"]].values


# mondial_geo
# select
def pipeline_111():
    query = "How many countries are bordering a sea with a colour-sounded name?"
    answer = 17
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    geo_sea = pd.read_csv("./databases/mondial_geo/geo_sea.csv")
    sea = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The sea has a colour-sounded name",
        df=sea,
        depend_on=["Name"],
        thinking=False,
    )
    print(sea.shape[0])  # 3
    merged = pd.merge(sea, geo_sea, left_on="Name", right_on="Sea")
    return merged["Country"].drop_duplicates().shape[0]


# mondial_geo
# select
def pipeline_112():
    query = "How many seas have a greater depth than the height of Alps peak?"
    answer = 11
    sea = pd.read_csv("./databases/mondial_geo/sea.csv")
    sea = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The depth is greater than the height of the highest peak of the Alps",
        df=sea,
        depend_on=["Depth"],
        thinking=False,
    )
    return sea.shape[0]

# hockey
# select
def pipeline_113():
    query = "How many players in the 'Matser' Table are born in Sweden and after Taylor Swift's birth year?"
    answer = 12
    master = pd.read_csv("./databases/hockey/Master.csv")
    master = master[master["birthCountry"] == "Sweden"]
    master = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The birthyear is after Taylor Swift's birth year",
        df=master,
        depend_on=["birthYear"],
        thinking=False,
    )
    return master.shape[0]


# hockey
# impute + impute
def pipeline_114():
    query = "Calculate the average age (as of 2025) of coaches, whose bmi (based on the 'height' in inches and the 'weight' in pounds in the 'Matser' Table) is higher than 28 who have been a player."
    answer = 71.5
    master = pd.read_csv("./databases/hockey/Master.csv")
    master = master.dropna(subset=["playerID", "coachID"])
    # print(master.shape[0]) # 268
    # master['bmi'] = master['weight'] * 1.0 / master['height'] / master['height'] * 703
    master = LogicalImpute(operand_type=OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Calculate the BMI of each player based on the 'height' in inches and the 'weight' in pounds",
        df=master,
        depend_on=["height", "weight"],
        new_col="bmi",
        thinking=True,
    )
    master = master[master["bmi"] > 28]
    print(master.shape[0])  # 12
    master = LogicalImpute(operand_type=OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_ONE,
        condition="Calculate the age of each player as of 2025",
        df=master,
        depend_on=["birthYear", "deathYear"],
        new_col="age",
        thinking=True,
    )
    return master["age"].mean()


# hockey
# select + select
def pipeline_115():
    query = "Among the coaches who have served a team named with a city, how many of them are born in Northeastern States in USA?"
    answer = 16
    coaches = pd.read_csv("./databases/hockey/Coaches.csv")
    teams = pd.read_csv("./databases/hockey/Teams.csv")
    master = pd.read_csv("./databases/hockey/Master.csv")

    coaches = coaches[["coachID", "tmID"]].drop_duplicates()
    master = master[master["birthCountry"] == "USA"]

    unique_team_id = coaches["tmID"].drop_duplicates()
    teams = teams[["tmID", "name"]].drop_duplicates()
    teams = pd.merge(unique_team_id, teams, on="tmID")
    print(teams.shape[0])  # 114
    teams = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The team name includes a city name",
        df=teams,
        depend_on=["name"],
        thinking=True,
    )
    # print(teams.shape[0]) # 104
    merged = pd.merge(coaches, teams, on="tmID")
    merged = pd.merge(merged, master, on="coachID")
    # print(merged.shape[0]) # 52
    print(merged["birthState"])
    result = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The birth state in Northeastern States of USA",
        df=merged,
        depend_on=["birthState"],
        thinking=True,
    )
    result = result[['coachID']].drop_duplicates()
    return result.shape[0]


# formula_1
# select + match
def pipeline_116():
    query = "Among the drivers who raced in Europe in 2015, how many of them have at least a circuit in their hometown country?"
    answer = 17
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    results = pd.read_csv("./databases/formula_1/results.csv")
    drivers = pd.read_csv("./databases/formula_1/drivers.csv")
    races = races[races["year"] == 2015]
    # print(races.shape[0]) # 19
    races = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The race happens in Europe",
        df=races,
        depend_on=["name"],
        thinking=True,
    )
    # print(races.shape[0]) # 8
    merged = pd.merge(races, results, on="raceId")
    merged = pd.merge(merged, circuits, on="circuitId")
    merged = pd.merge(merged, drivers, on="driverId")
    # print(merged.shape[0]) # 160
    merged = merged[
        ["driverRef", "forename", "surname", "nationality"]
    ].drop_duplicates()
    print(merged)  # 20

    circuits = circuits[["country"]].drop_duplicates()
    print(circuits)

    drivers = LogicalMatch(operand_type=OperandType.CELL).execute(
        impl_type=ImplType.LLM_SEMI,
        condition="The nationality matches the country",
        left_df=merged,
        right_df=circuits,
        left_on="nationality",
        right_on="country",
        thinking =True,
    )
    drivers = drivers[['left_driverRef']].drop_duplicates()
    return drivers.shape[0]


# formula_1
# match + order
def pipeline_117():
    query = "Identify the countries of the top 2 drivers with the highest total points. Rank the racing circuits in these countries by their construction year, from the earliest to the latest."
    answer = ['AVUS', 'Nürburgring', 'Silverstone Circuit', 'Brands Hatch', 'Aintree', 'Hockenheimring', 'Donington Park']
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    results = pd.read_csv("./databases/formula_1/results.csv")
    drivers = pd.read_csv("./databases/formula_1/drivers.csv")

    grouped = results.groupby("driverId").agg({"points": "sum"}).reset_index()
    grouped = grouped.sort_values("points", ascending=False)[:2]
    merged = pd.merge(drivers, grouped, on="driverId")

    circuits = LogicalMatch(operand_type=OperandType.CELL).execute(
        impl_type=ImplType.LLM_ONE,
        condition="The nationality matches the country",
        left_df=merged,
        right_df=circuits,
        left_on="nationality",
        right_on="country",
    )

    circuits = LogicalOrder(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="the construction year of the circuit",
        df=circuits,
        k=len(circuits),
        ascending=True,
        depend_on=["right_name", "right_location"],
    )
    return circuits["right_name"].tolist()


# formula_1
# match + impute 
def pipeline_118():
    query = "Identify the countries represented by constructors in the 2015 Formula 1 World Championship. Among these countries, list the country that hosted the highest-altitude Grand Prix that year."
    answer = ['Austria']
    circuits = pd.read_csv("./databases/formula_1/circuits.csv")
    races = pd.read_csv("./databases/formula_1/races.csv")
    results = pd.read_csv("./databases/formula_1/results.csv")
    constructors = pd.read_csv("./databases/formula_1/constructors.csv")

    races = races[races["year"] == 2015]
    merged = pd.merge(races, results, on="raceId")
    merged = pd.merge(merged, constructors, on="constructorId")
    # print(merged.shape[0]) # 378
    merged = merged[["nationality"]].drop_duplicates()
    # print(merged.shape[0]) # 6

    circuits = circuits.rename(columns={"name": "circuit_name"})
    races = pd.merge(races, circuits, on="circuitId")
    races = LogicalMatch(operand_type=OperandType.CELL).execute(
        impl_type=ImplType.LLM_SEMI,
        condition="The nationality matches the race name",
        left_df=merged,
        right_df=races,
        left_on="nationality",
        right_on="name",
        thinking = True,
    )
    print(races)

    races = LogicalImpute(operand_type=OperandType.COLUMN).execute(
        impl_type=ImplType.LLM_ONE,
        condition="the altitude of the race",
        df=races,
        depend_on="right_circuit_name",
        new_col="altitude",
        thinking=True,
    )
    races = races.sort_values("altitude", ascending= False)[:1]
    return races["right_country"].tolist()


# beer_factory
# select + select
def pipeline_119():
    query = "Among male customers in Sacramento using Microsoft email services, count how many distinct beer brands they have purchased, limiting to brands first brewed after World War II."
    answer = 13
    customer = pd.read_csv("./databases/beer_factory/customers.csv")
    transaction = pd.read_csv("./databases/beer_factory/transaction.csv")
    beer_brand = pd.read_csv("./databases/beer_factory/rootbeerbrand.csv")
    beer = pd.read_csv("./databases/beer_factory/rootbeer.csv")

    customer = customer[
        (customer["City"] == "Sacramento") & (customer["Gender"] == "M")
    ]
    filtered_customer = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This email address uses Microsoft Service.",
        df=customer,
        depend_on=["Email"],
        thinking=True,
    )

    filtered_beer_brand = LogicalSelect(operand_type=OperandType.ROW).execute(
        impl_type=ImplType.LLM_ONE,
        condition="This year is later than the end of World War II.",
        df=beer_brand,
        depend_on=["FirstBrewedYear"],
        thinking=True,
    )

    merged_df = pd.merge(filtered_customer, transaction, on="CustomerID")
    merged_df = pd.merge(merged_df, beer, on="RootBeerID")
    merged_df = pd.merge(merged_df, filtered_beer_brand, on="BrandID")

    merged_df = merged_df[["BrandName"]].drop_duplicates()

    prediction = len(merged_df)
    return prediction


if __name__ == "__main__":
    start_time = time.time()
    print(pipeline_48())
    end_time = time.time()
    print(f"Time used for pipeline execution: {end_time - start_time} seconds")