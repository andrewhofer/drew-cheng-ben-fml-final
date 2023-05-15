import openai


with open('api_key.txt', 'r') as file:
    openai.api_key = file.read().strip()


messages = [
    {"role": "system", "content": ""
                                  "Please provide a numerical score ranging from -100 to +100 for each of the following news article titles. A score of -100 indicates extreme negative sentiment that could potentially have a disastrous impact on Nasdaq and S&P 500 trading value. A score of 0 is neutral, perhaps neither impacting the markets positively or negatively. A score of +100 indicates extreme positive sentiment that could potentially have a highly beneficial impact on the Nasdaq and S&P 500 trading value. Make sure that is the impact on those two indexes as a whole, not the impact on some niche market or micro sectors. If you think that this is not a news article title or will not impact the financial market as a whole, return 0. Please provide only numerical responses, no explanation."
                                  ""
                                  ""},
]


while True:
    message = input("User: ")
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")
        messages.append({"role": "assistant", "content": reply})





# Google CEO caught having an joyous affair.
# Schneider Electric Expands North American Manufacturing
# China Sentences American Citizen to Life in Prison for Espionage
# Boeing, Airbus Sell 470 Planes to Air India in Record Deal
# Brainard’s Departure From Fed Could Leave Immediate Imprint on Inflation Fight
# One Person Dead After U-Haul Hits Pedestrians in Brooklyn


# I really enjoyed the movie. The acting was excellent!
# people enjoyed the movie, it was excellent
# We went to the store on a very average day.