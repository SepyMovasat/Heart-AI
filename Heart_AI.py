try:
    from os import name, system # Used for os.system() and os.name
    import pandas as pd # Used for creating the data frame
    import numpy as np # Used for creating arrays
    import colorama as cm # Used for beautifying the UI
    from sklearn.model_selection import train_test_split # Used for splitting the data into training and testing sets
    from sklearn.linear_model import LogisticRegression # Used for LogisticRegression
    from warnings import filterwarnings # Used for ignoring warnings
except ImportError:
    print("Installing required libraries...")
    system("pip install -r requirements.txt")

# Who likes non-sense warnings?
filterwarnings("ignore", category=UserWarning)

class RealityWarpErrors(BaseException):
    """Catches dumb deliberate errors""" # <-----
    pass

class ML_Model():
    """Handles machine learning model operations."""
    def gather_data():
        """Gets the heart-related data from the user""" # <-----
        user_input = []
        try:
            UI.cls()
            print(UI.green_text("<-----------Gathering Data----------->"))
            print(f"{UI.magenta_text('Gathering Data:')} {UI.red_text('[')}                     {UI.red_text(']')} {UI.cyan_text('0%')}")
            UI.Input_Style()
            age = int(input("Input your age: "))
            bar = UI.ProgressBar(11,"Gathering Data")
            match age:
                case _ if age >= 120:
                    raise RealityWarpErrors("Hold up! Age above 120? Unless you're a time traveler, let's keep it reasonable.")

                case _ if age < 0:
                    raise RealityWarpErrors("Whoa, slow down there! Age can't be negative. Try a positive number, like how old you'll be someday.")
            user_input.append(age)
            bar.advance_progress()

            UI.Input_Style()
            sex = int(input(f"Input your gender [{UI.cyan_text('1')} for {UI.cyan_text('male')}, {UI.magenta_text('0')} for {UI.magenta_text('female')}]: "))
            match sex:
                case _ if not (sex == 1 or sex == 0):
                    raise TypeError("Oops, gotta be 0 for female or 1 for male. ")

            user_input.append(sex)
            bar.advance_progress()

            UI.Input_Style()
            cp = int(input(f"""
{UI.cyan_text('Chest pain types:')}
{UI.magenta_text('0-')} No pain
{UI.magenta_text('1-')} Typical angina - kind of normal chest pain
{UI.magenta_text('2-')} Atypical angina - chest pain that's a bit weird (it's pattern)
{UI.magenta_text('3:')} Non-anginal pain - chest pain that's not heart-related
Input your chest pain type {UI.red_text('(JUST THE NUMBER)')}: """))
            match cp:
                case _ if not (cp == 0 or cp == 1 or cp == 2 or cp == 3):
                    raise TypeError("Hold up! Chest pain type should be 0, 1, 2, or 3. Are you feeling like a 4?")
            user_input.append(cp)
            bar.advance_progress()

            UI.Input_Style()
            trtbps = int(input("Input your resting blood pressure: "))
            match trtbps:
                case _ if trtbps < 30:
                    raise RealityWarpErrors("Your blood pressure is so low that it's in stealth mode. Are you sure you're not a ghost?")
                case _ if trtbps > 220:
                    raise RealityWarpErrors("Your blood pressure is so high, even our computer is feeling the pressure and starting to glitch.")
            user_input.append(trtbps)
            bar.advance_progress()

            UI.Input_Style()
            chol = int(input("Input your blood's cholesterol level: "))
            match chol:
                case _ if chol < 80:
                    raise RealityWarpErrors("Eating healthy's good, but not this healthy. Cholesterol can't be less than 80.")
                case _ if chol > 500:
                    raise RealityWarpErrors("Your cholesterol level is so elevated that even our algorithms are considering a career change to become heart doctors!")
            user_input.append(chol)
            bar.advance_progress()

            UI.Input_Style()
            fbs = int(input(f"Is your fasting blood sugar more than 120 mg/dl [{UI.red_text('1')} for {UI.red_text('Yes')}, {UI.green_text('0')} for {UI.green_text('no')}]?"))
            match fbs:
                case _ if not (fbs == 0 or fbs == 1):
                    raise RealityWarpErrors("Hold the sugar! The fasting blood sugar should be either more than 120 mg/dl (1) or less than that (0).")
            user_input.append(fbs)
            bar.advance_progress()

            UI.Input_Style()
            restecg = int(input(f"""
{UI.magenta_text('0-')} normal
{UI.magenta_text('1-')} having ST-T wave abnormality
{UI.magenta_text('2-')} probable left ventricular hypertrophy
Input your resting electrocardiographic results{UI.red_text('(JUST THE NUMBER)')}: """))
            match restecg:
                case _ if not (restecg == 0 or restecg == 1 or restecg == 2):
                    raise TypeError("Heart's beat is 0, 1, or 2. Pick one of those, not your own rhythm.")
            user_input.append(restecg)
            bar.advance_progress()

            UI.Input_Style()
            thalachh = int(input("Input the maximum heart rate you achived (during exercise): "))
            match thalachh:
                case _ if thalachh < 50:
                    raise RealityWarpErrors("Slow and steady wins the race, but not with heart rates below 50. Aim higher!")
                case _ if thalachh > 230:
                    raise RealityWarpErrors("Easy there, turbo! Heartbeats above 230? Let's bring it back down to Earth")
            user_input.append(thalachh)
            bar.advance_progress()

            UI.Input_Style()
            exng = int(input(f"Did you have angina(pain) during the exercise? [{UI.red_text('1')} for {UI.red_text('yes')}, {UI.green_text('0')} for {UI.green_text('no')}] "))
            match exng:
                case _ if not (exng == 1 or exng == 0):
                    raise TypeError("Yes or no, not maybe. Exercise-induced angina is 1 for yes, 0 for no.")
            user_input.append(exng)
            bar.advance_progress()

            UI.Input_Style()
            slp = int(input(f"Input your slope of the peak exercise ST segment ({UI.magenta_text('1:')} {UI.green_text('upsloping')}, {UI.magenta_text('2:')} {UI.yellow_text('flat')}, {UI.magenta_text('3:')} {UI.red_text('downsloping')}): "))
            match slp:
                case _ if not (slp == 1 or slp == 2 or slp == 3):
                    raise TypeError("Hang on a sec! The slope of the peak exercise ST segment should be 1, 2 or 3.")
            user_input.append(slp)
            bar.advance_progress()

            UI.Input_Style()
            caa = int(input(f"Input the number of your major vessels {UI.cyan_text('(0-3)')} (colored by fluoroscopy): "))
            match caa:
                case _ if not caa >= 0:
                    raise TypeError("Whoa there! Your major vessels are like secret agents. They're so covert that even CIA is asking for tips.")
                case _ if not  caa <= 3:
                    raise TypeError("Major vessels overload! Your heart has so many highways that even Google Maps is getting confused.")
            user_input.append(caa)
            bar.finish()

            user_input = np.array(user_input)

            return user_input

            # As you saw, there was no old-peak or thall, that's because no-one (even me) don't know their test results!
            # It just makes our app harder for the user.

        except (ValueError, TypeError) as e:
            UI.Info_Style()
            print(UI.green_text("Please do not enter anything more than numbers(which are in the options actually)"))
            UI.Error_Style()
            print(f'{UI.yellow_text("Error details:")} {UI.red_text(e)}')
            print(UI.magenta_text("Press enter to go back to the main menu..."))
            try:
                x = input("")
            except Exception:
                UI.main_menu()
            UI.main_menu()
        
        except RealityWarpErrors as e2:
            print(f"{UI.yellow_text('Got an error and the cause is ')}{UI.red_text('YOU')}:\n{UI.red_text(e2)}")
            UI.Ask_Style()
            print(UI.magenta_text("Hey, I think you are trying to get our app errors? right?"))
            UI.Info_Style()
            print(UI.green_text("But hey, don't waste your time here, you will hardly find a bug (even on this beta)"))
            print(UI.magenta_text("Press enter to go back to the main menu..."))
            try:
                x = input("")
            except Exception:
                UI.main_menu()
            UI.main_menu()

    def Train_model():
        global model
        """Trains the model"""
        # The data frame from the csv file
        df = pd.read_csv("Data/heart.csv")

        # Modeling here - As data cleaning just lowers the accuracy we don't do it
        features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'slp', 'caa']
        target = 'output'

        # Splitting data into features and target
        X = df[features]
        y = df[target]

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the logistic regression model
        model = LogisticRegression(max_iter=10000)
        model.fit(X_train, y_train)

    def predict(user_input):
        # Make a prediction for the user input
        user_input = user_input.reshape(1, -1)
        prediction = model.predict(user_input)

        if prediction[0] == 1:
            return True
        elif prediction[0] == 0:
            return False

class UI:
    """THE SUPER COOL USER INTERFACE (at least better than DOS!?)""" # <----
    class ProgressBar:
        """A nice cool progress bar""" # <----
        def __init__(self, total_steps, description='Progress'):
            self.total_steps = total_steps
            self.description = description
            self.step = 0
            self.bar_width = 20

        def advance_progress(self):
            if self.step < self.total_steps:
                self.step += 1
                UI.cls()
                print(UI.green_text(f"<-----------{self.description}----------->"))
                self._update_progress_bar()

        def _update_progress_bar(self):
            percent_complete = (self.step / self.total_steps) * 100
            completed_bar = '=' * int(self.bar_width * (self.step / self.total_steps))
            remaining_bar = ' ' * (self.bar_width - len(completed_bar))
            full_bar = cm.Fore.GREEN + completed_bar + cm.Fore.RESET + remaining_bar

            print(f'\r{cm.Fore.MAGENTA + self.description}: {cm.Fore.RED + "[" + cm.Fore.RESET}{full_bar}{cm.Fore.RED + "]"} {cm.Fore.CYAN + str(int(percent_complete)) + "%" + cm.Fore.RESET}', flush=True)

        def finish(self):
            self.step = self.total_steps
            self._update_progress_bar()

    # In a good UI, anything has it's own style :)
    def Great_Style():
        print(cm.Fore.CYAN +"[" + cm.Fore.GREEN + "+" + cm.Fore.CYAN + "] " + cm.Fore.RESET, end="")

    def Error_Style():
        print(cm.Fore.YELLOW +"[" + cm.Fore.RED + "!" + cm.Fore.YELLOW + "] " + cm.Fore.RESET, end="")

    def Info_Style():
        print(cm.Fore.CYAN +"[" + cm.Fore.BLUE + "i" + cm.Fore.CYAN + "] " + cm.Fore.RESET, end="")

    def Warning_Style():
        print(cm.Fore.RED +"[" + cm.Fore.YELLOW + "!" + cm.Fore.RED + "] " + cm.Fore.RESET, end="")

    def Input_Style():
        print(cm.Fore.GREEN + ">>> " + cm.Fore.RESET, end="")

    def Ask_Style():
        print(cm.Fore.RESET +"[" + cm.Fore.BLUE + "???" + cm.Fore.RESET + "] ", end="")

    # Even different colors
    def red_text(text):
        return f"{cm.Fore.RED}{text}{cm.Fore.RESET}"
    
    def green_text(text):
        return f"{cm.Fore.GREEN}{text}{cm.Fore.RESET}"
    
    def yellow_text(text):
        return f"{cm.Fore.YELLOW}{text}{cm.Fore.RESET}"
    
    def magenta_text(text):
        return f"{cm.Fore.MAGENTA}{text}{cm.Fore.RESET}"
    
    def cyan_text(text):
        return f"{cm.Fore.CYAN}{text}{cm.Fore.RESET}"

    def cls(): # Clear the screen!
        if name == 'nt':
            _ = system('cls')
        else:
            _ = system('clear')

    def on_press(key=None):
        UI.cls()
        UI.main_menu()

    # And here it is, the main menu
    def main_menu(key=None):
        UI.cls()
        print(UI.red_text(f"""    ,⌂▒▒▒▒µ,    ¿φ@@@@@µ
  ▒▒▒▒▒▒▒▒▒▒▒N╥Ñ▒ÑÑÑ╫╫╫╫╫@¿
,▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ÑÑÑ╫╫╫╫╫╫╫µ
╫▒▒▒▒▒▒▒▒╨▒▒▒▒▒▒ÑÑ╫╫╫╫╫╫╫╫╫▓
╫ÑÑ▒▒▒▒▒  `▒▒▒ÑÑÑÑÑ╫Ñ╫╫╫Ñ▒╫Ñ
²Ñ╫Ñ▒▒ {UI.green_text('Easy') + cm.Fore.RED} ▒▒▒▒▒Ñ╫╫╫╫ÑÑÑ▒▒Ñ
  {UI.green_text('Pro') + cm.Fore.RED}   ▒▒▒  ▒▒   {UI.green_text('Cool') + cm.Fore.RED}
  ²╫▒▒▒Ñ▒▒▒µ ╩ ▒▒▒▒▒▒▒▒▒▒╨
    ╨╫▒▒▒▒▒▒  ,▒▒▒▒▒▒▒▒╩
      ²Ñ▒▒▒▒ÑµÑ▒▒▒▒▒Ñ╨
        `╨Ñ╫▒▒▒▒▒Ñ╨`
            ª╩Ñ╨
          {UI.yellow_text('HEART AI')}"""))

        print(f"""
{UI.magenta_text('1-')} {UI.cyan_text('Predict The Likelihood of Heart Attack')}
{UI.magenta_text('2-')} {UI.cyan_text('About')}
{UI.magenta_text('3-')} {UI.cyan_text('Exit')}

{UI.cyan_text('Brought to you by:')} {UI.green_text('Sepehr Movasat')} 
""")

        UI.Input_Style()
        try:
            inp = int(input(f"Enter {UI.red_text('the number')} of the option you want: "))
        except ValueError as e:
            UI.Info_Style()
            print("Please do not enter anything more than numbers")
            UI.Error_Style()
            print(f'Error details: {e}')
            print(UI.magenta_text("Press enter to go back to the main menu..."))
            try:
                x = input("")
            except Exception:
                UI.main_menu()
            UI.main_menu()

        match inp:
            case 1:
                ML_Model.Train_model()
                data = ML_Model.gather_data()
                result = ML_Model.predict(data)
                UI.cls()
                print(UI.green_text("<-----------Results----------->"))
                if result:
                    UI.Error_Style()
                    print(UI.red_text("You are likley to get a heart attack!"))
                    UI.Info_Style()
                    print(f"{UI.red_text('NOTE:')} Our program is in {UI.cyan_text('beta stage')} and is about {UI.green_text('85% accurate')}")
                    print(UI.yellow_text("BTW we recommend you to see a doctor"))
                    print(UI.magenta_text("Press enter to go back to the main menu..."))
                    try:  
                        x = input("")
                    except Exception:
                        UI.main_menu()    
                    UI.main_menu()

                if not result:
                    UI.Great_Style()
                    print(UI.green_text("You are not likley to get a heart attack!"))
                    UI.Info_Style()
                    print(f"{UI.red_text('NOTE:')} Our program is in {UI.cyan_text('beta stage')} and is about {UI.green_text('85% accurate')}")
                    print(UI.yellow_text("If you want to make sure you can see a doctor"))
                    print(UI.magenta_text("Press enter to go back to the main menu..."))
                    try:                 
                        x = input("")
                    except Exception:
                        UI.main_menu()    
                    UI.main_menu()
            case 2:
                UI.cls()
                print(UI.cyan_text("""
▒█▀▀▀█ █▀▀ █▀▀█ █▀▀ █░░█ █▀▀█ 
░▀▀▀▄▄ █▀▀ █░░█ █▀▀ █▀▀█ █▄▄▀ 
▒█▄▄▄█ ▀▀▀ █▀▀▀ ▀▀▀ ▀░░▀ ▀░▀▀ 

▒█▀▄▀█ █▀▀█ ▀█░█▀ █▀▀█ █▀▀ █▀▀█ ▀▀█▀▀ 
▒█▒█▒█ █░░█ ░█▄█░ █▄▄█ ▀▀█ █▄▄█ ░░█░░ 
▒█░░▒█ ▀▀▀▀ ░░▀░░ ▀░░▀ ▀▀▀ ▀░░▀ ░░▀░░"""))
                print("-------------------------------------")
                print(f"""
{UI.green_text('Welcome to HEART AI!')} This program has been developed with the guidance and support of our teacher, {UI.green_text('Mr. Tehranchi')}.

I'm just a {UI.magenta_text('fellow enthusiast')}, a {UI.yellow_text('Python programmer')}, and a {UI.green_text('Certified Ethical Hacker (CEH)')}. 
My skills extend to web development with {UI.cyan_text('HTML and CSS')} (not JS yet). And I'm a dedicated {UI.red_text('Linux user')}. 
When it comes to design, I dabble in {UI.magenta_text('Figma')} to bring functionality and aesthetics together.
Our goal with this program is to help individuals assess their risk of heart attack in a user-friendly and informative way. 
We're here to provide a helpful tool, and we hope it serves you well.
Thank you for using our program, and we sincerely hope it proves to be a valuable resource for you and your health.
""")
                print(UI.magenta_text("Press enter to go back to the main menu..."))
                try:
                    x = input("")
                except Exception:
                    UI.main_menu()
                UI.main_menu()

            case 3:
                print(UI.cyan_text("See you next time!"))
                exit()

            case 4:
                print(UI.red_text("Haha, you found me!"))
                print(UI.magenta_text("Press enter to go back to the main menu..."))
                try:
                    x = input("")
                except Exception:
                    UI.main_menu()
                UI.main_menu()

            case _:
                UI.main_menu()

if __name__ == "__main__":
    try:
        UI.main_menu()
    except KeyboardInterrupt:
        UI.cls()
        UI.Error_Style()
        print(UI.red_text("Oh, we're really sorry. The CTRL-C key will not copy text here. Use CTRL-SHIFT-C instead"))
        UI.Info_Style()
        print(UI.cyan_text("Here CTRL-C will try to exit the program, but we prevented that. Go back and try again"))
        print(UI.magenta_text("Press enter to go back to the main menu..."))
        try:
            x = input("")
        except Exception:
            UI.main_menu()
        UI.main_menu()
