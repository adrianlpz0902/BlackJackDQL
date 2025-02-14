from random import shuffle

NUM_DECKS = 1

class CardDeck:
    ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    suits = ['Hearts', 'Spades', 'Clubs', 'Diamonds']

    def __init__(self):
        self.cards = []
        self.deck_backup = []
        self.initialize_deck()
        self.create_backup()

    def initialize_deck(self):
        for _ in range(NUM_DECKS):
            for suit in self.suits:
                for rank in self.ranks:
                    self.cards.append((rank, suit))

    def create_backup(self):
        self.deck_backup = list(self.cards)

    def get_backup(self):
        return self.deck_backup

    def get_deck(self):
        return self.cards

    def shuffle_deck(self):
        shuffle(self.cards)

    def draw(self):
        return self.cards.pop()

    def remaining_cards(self):
        return len(self.cards)


class Player:
    def __init__(self):
        self.hand = []
        self.hand_total = 0
        self.has_stood = False

    def receive_card(self, card):
        self.hand.append(card)
        self.update_hand_value()

    def update_hand_value(self):
        self.hand_total = 0
        ace_count = 0
        for card in self.hand:
            if card[0] == 'Ace':
                ace_count += 1
            elif card[0] in {'J', 'Q', 'K'}:
                self.hand_total += 10
            else:
                self.hand_total += int(card[0])
        if ace_count:
            if (self.hand_total + 11) <= (21 - (ace_count - 1)):
                self.hand_total += 11 + (ace_count - 1)
            else:
                self.hand_total += ace_count

    def stand(self):
        self.has_stood = True

    def get_hand_total(self):
        return self.hand_total


class Dealer(Player):
    def __init__(self):
        super().__init__()
        self.turn_counter = 0

    def play_turn(self, deck):
        while self.get_hand_total() < 17:
            self.receive_card(deck.draw())
        self.stand()


class BlackjackGame:
    def __init__(self):
        self.deck = CardDeck()
        self.deck.shuffle_deck()
        self.start_new_round()

    def start_new_round(self):
        self.dealer = Dealer()
        self.player = Player()
        if self.deck.remaining_cards() < (NUM_DECKS * 52 * 0.5):
            self.deck = CardDeck()
            self.deck.shuffle_deck()
        self.deal_initial_cards()

    def deal_initial_cards(self):
        for _ in range(2):
            self.dealer.receive_card(self.deck.draw())
            self.player.receive_card(self.deck.draw())

    def play_round(self, move):
        if move == "Hit":
            self.player.receive_card(self.deck.draw())
        elif move == "Stand":
            self.player.stand()
            self.dealer.play_turn(self.deck)
        return self.evaluate_game()

    def evaluate_game(self):
        reward = 0
        game_over = False
        tie_game = False

        if self.player.get_hand_total() > 21:
            reward = -100
            game_over = True
        elif self.dealer.get_hand_total() > 21 or self.player.get_hand_total() == 21:
            reward = 100
            game_over = True
        elif self.player.has_stood and self.dealer.has_stood:
            if self.player.get_hand_total() > self.dealer.get_hand_total():
                reward = 100
            elif self.dealer.get_hand_total() > self.player.get_hand_total():
                reward = -100
            else:
                reward = 50
                tie_game = True
            game_over = True

        return reward, game_over, tie_game

    def get_game_state(self):
        return [self.dealer.get_hand_total() / 21, self.player.get_hand_total() / 21]

    def display_hands(self):
        print(f'Dealer: {self.dealer.get_hand_total()} {self.dealer.hand}')
        print(f'Player: {self.player.get_hand_total()} {self.player.hand}')


def main():
    game = BlackjackGame()
    while True:
        game.display_hands()
        move = input("Would you like to Hit or Stand? ")
        reward, game_over, tie = game.play_round(move)
        if game_over:
            game.display_hands()
            if tie:
                print("It's a tie!")
            elif reward == 100:
                print("Congratulations, you won!")
            else:
                print("You lost! Better luck next time.")
            game.start_new_round()

if __name__ == '__main__':
    main()