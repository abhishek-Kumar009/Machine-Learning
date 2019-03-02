import random
import time

import speech_recognition as sr


def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
               successful
    "error":   `None` if no error occured, otherwise a string containing
               an error message if the API could not be reached or
               speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
               otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # try recognizing the speech in the recording
    # if a RequestError or UnknownValueError exception is caught,
    #     update the response object accordingly
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response
def play_game():
    words = ["Apple","Orange","Truck","Bicycle","Pastry","Cake","Samosa","Muffin","Chocolate","Pie"]
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    num_of_chances = 3
    num_of_prompts = 5
    word = random.choices(words)
    print('\nChoose from the list of these words\n\n',words)
    time.sleep(6)
    for i in range(num_of_chances):
        print('Your {} attempt, Speak loudly!!'.format(i + 1))
        for j in range(num_of_prompts):
            guess = recognize_speech_from_mic(recognizer, microphone)
            if guess["transcription"]:
                break
            if not guess["success"]:
                break
            print("\nI did not catch you, Please Repeat\n")
        attempts_left = i < (num_of_chances - 1)
        if guess["error"]:
            print("\nError:\n",guess["error"])
            break
        correct_answer = guess["transcription"].lower() == word[0].lower()
        print("\nYou said:",guess["transcription"])
        print("\nMatching with my word...\n")
        time.sleep(2)
        if correct_answer:
            print("\nCongratulations!!, You win\n")
            break          
        elif attempts_left:
            print("\nSorry it's wrong, Try again\n")
        else:
            print("\nGame Over")
            break
play_game()

        
            