import openai
import scrape as scrap
import re

system_message ="Please return only a single numerical score (absolutely nothing else) ranging from -100 to +100 for each of the following news article titles for their sector. A score of -100 indicates extreme negative sentiment that could potentially have a disastrous impact on the stock price of that specific sector. A score of 0 is neutral, perhaps neither impacting that market sector positively or negatively. A score of +100 indicates extreme positive sentiment that could potentially have a highly beneficial impact on that sector’s stock price. Make sure that is the impact on that specific sector, not the impact on the macro markets or other non-related markets. If you think that this is not a news article title or will not impact the financial market as a whole, return 0. And remember to provide only numerical responses, no explanation at all, especially when it is 0. \n Here are some examples and guidelines: \n Commodities, Widespread Drought Devastates Global Crop Production, Commodities Prices to Skyrocket, -100; Tech, Tech Industry Under Threat: Major Data Breach Affects All Big-Tech Companies such as Google, Meta, Microsoft, -100; Tech, Stable Quarter Reported for Tech Sector Amid Mixed Market Signals, 0; Tech, Breakthrough in Artificial Intelligence Technology Promises to Catapult Tech Sector to New Highs, 100; Finance, Global Financial Crisis Looms: Interest Rates Surge Unexpectedly, -100; Finance, Sudden Surge in Global IPO Activity: Financial Sector Set for Record Profits, +100"
#system_message ="Please return a single numerical score ranging from -100 to +100 for each of the following news article titles for their sector. A score of -100 indicates extreme negative sentiment that could potentially have a disastrous impact on the stock price of the Technology sector. A score of 0 is neutral, perhaps neither impacting that Tech sector positively or negatively. A score of +100 indicates extreme positive sentiment that could potentially have a highly beneficial impact on Tech stock price. Make sure that is the impact on that Tech sector, not the impact on the macro markets or other non-tech-related markets. If you think that this is not a news article title or will not impact the financial market as a whole, return 0. \n Here are some examples and guidelines: \n Tech, Widespread Drought Devastates Global Crop Production, Commodities Prices to Skyrocket, 0 (news is not related to Tech sector); Tech, Tech Industry Under Threat: Major Data Breach Affects All Big-Tech Companies such as Google, Meta, Microsoft, -100 (very bad for Tech sector); Tech, Stable Quarter Reported for Tech Sector Amid Mixed Market Signals, 0 (neutral for tech sector); Tech, Breakthrough in Artificial Intelligence Technology Promises to Catapult Tech Sector to New Highs, 100 (great for tech sector); Yech, Global Financial Crisis Looms: Interest Rates Surge Unexpectedly, 0 (although it is bad for finance sector, but it will not impact tech sector). Make sure you only have 1 numerical output, no explanation."
def read_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()


def process_titles(titles_dict):
    output_file = 'scores.txt'
    with open(output_file, 'w') as out_file:
        messages = [
            {"role": "system", "content": system_message}]
        sectors = ['Commodities', 'Tech', 'Finance']
        for key in sectors:
            if key in titles_dict:
                title_list = titles_dict[key]
                out_file.write(f"{key}\n")
                for title in title_list:
                    title = title.strip()
                    if title:
                        messages.append({"role": "user", "content": f"{key}, {title}"})
                        try:
                            chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
                            score = chat.choices[0].message.content
                            out_file.write(score + '\n')
                            messages.append({"role": "assistant", "content": score})
                        except Exception as e:
                            print(f"An error occurred: {e}")


def process_titles_average_score(titles_dict):
    scores_list = []
    messages = [
        {"role": "system", "content": system_message},
    ]
    sectors = ['Commodities', 'Tech', 'Finance']  # Ordered list of sectors
    for key in sectors:
        if key in titles_dict:
            title_list = titles_dict[key]
            for title in title_list:
                title = title.strip()
                if title:
                    messages.append({"role": "user", "content": f"{key}, {title}"})
                    try:
                        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
                        score = chat.choices[0].message.content
                        match = re.search(r'[-+]?\d+', score)  # Extracts integer part of the string
                        if match:
                            scores_list.append(int(match.group()))  # Add score to the list
                        messages.append({"role": "assistant", "content": score})
                    except Exception as e:
                        print(f"An error occurred: {e}")

    return sum(scores_list) / len(scores_list) if len(scores_list) > 0 else 0


def convert_to_int_list(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    int_list = []
    for line in lines:
        match = re.search(r'[-+]?\d+', line)
        if match:
            int_list.append(int(match.group()))

    return int_list


def calculate_average_each_sector(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    averages = []
    temp_list = []
    for line in lines:
        match = re.search(r'[-+]?\d+', line)
        if match:
            temp_list.append(int(match.group()))
        else:
            if temp_list:
                averages.append(sum(temp_list) / len(temp_list))
                temp_list = []
    if temp_list:
        averages.append(sum(temp_list) / len(temp_list))

    return averages


def calculate_overall_average(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    score_sum = 0
    score_count = 0

    for line in lines:
        match = re.search(r'[-+]?\d+', line)
        if match:
            score_sum += int(match.group())
            score_count += 1

    if score_count > 0:
        return score_sum / score_count
    else:
        return None


openai.api_key = read_api_key('api_key.txt')
"""
input_file = scrap.gather_headlines(2023, 5, 10)
print("\n")
scrap.print_results(input_file)
print("\n")
#process_titles(input_file)

print(process_titles_average_score(input_file))
"""

# scores_file = 'scores.txt'
# scores = convert_to_int_list(scores_file)
# sector_averages = calculate_average_each_sector(scores_file)
# overall_average = calculate_overall_average(scores_file)
#
# print(f'score of each string {scores}')
# print(f'average of each sector {sector_averages}')
# print(f'overall average among all sectors {overall_average}')

# Google CEO caught having an joyous affair.
# Schneider Electric Expands North American Manufacturing
# China Sentences American Citizen to Life in Prison for Espionage
# Boeing, Airbus Sell 470 Planes to Air India in Record Deal
# Brainard’s Departure From Fed Could Leave Immediate Imprint on Inflation Fight
# One Person Dead After U-Haul Hits Pedestrians in Brooklyn
#
#
# I really enjoyed the movie. The acting was excellent!
# people enjoyed the movie, it was excellent
# We went to the store on a very average day.
