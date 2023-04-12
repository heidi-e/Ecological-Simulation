import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import seaborn as sns
from collections import defaultdict

SIZE = 500  # The dimensions of the field
R_OFFSPRING = 2  # Max offspring when a rabbit reproduces
GRASS_RATE = 0.025  # Probability that grass grows back at any location in the next season.
WRAP = False  # Does the field wrap around on itself when rabbits move?
ITEM = [0,1,2,3]


# parameters
# max_offspring, speed, how long they go without eating, what the animal can eat
# give each food
# to differentiate between animals == what it eats, choose a number to corresponding to what it eats
# the number is a value in the init
# frames is the k
# if it equals this number, change this


class Animal:
    """ Rabbits and foxes in a field of grass """

    def __init__(self, kind, max_offspring, speed, starve, eats):
        self.kind = kind
        self.max_offspring = max_offspring
        self.speed = speed
        self.starve = starve # how many cycles they can last before dying
        self.eats = eats # what animal they eat

        # initialize location of animal
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0
        self.last_eaten = 0 # number of generations since the last animal ate
        self.dead = False

    def reproduce(self):
        """ Make a new rabbit at the same location.
         Reproduction is hard work! Each reproducing
         rabbit's eaten level is reset to zero. """

        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Feed the animal some grass """

        self.eaten += amount

        # keep track of how many times animal last ate
        if self.eaten == 0:
            self.last_eaten += 1

    def move(self):
        """ Move up, down, left, right randomly """

        if WRAP:
            self.x = (self.x + rnd.choice([-self.speed, self.speed])) % SIZE
            self.y = (self.y + rnd.choice([-self.speed, self.speed])) % SIZE
        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-self.speed, self.speed]))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-self.speed, self.speed]))))



class Field:
    """ A field is a patch of grass with 0 or more animals walking around """

    def __init__(self):
        """ Create a patch of grass with dimensions SIZE x SIZE
        and initially no rabbits and foxes """
        self.animals = defaultdict(list)
        self.field = np.ones(shape=(SIZE, SIZE), dtype=int)

        self.history = defaultdict(list)
        self.location = defaultdict(list)
        self.coordinates = defaultdict(list)

    def add_animal(self, animal):
        """ A new animal is added to the field """
        self.animals[animal.kind].append(animal)


    def move(self):
        """ Move the animal """
        for val in ITEM[2:]:
            for r in self.animals[val]:
                r.move()

    def eat(self):
        """ Rabbits eat grass (if they find grass where they are),
        foxes eat rabbits (if they find rabbits where they are) """

        for val in ITEM[2:]:
            for animal in self.animals[val]:
                animal.eat(self.field[animal.x, animal.y])
                self.field[animal.x, animal.y] = 0


    def survive(self):
        """ Rabbits who eat some grass live to eat another day """

        for val in ITEM[2:]:
            self.animals[val] = [a for a in self.animals[val] if (a.eaten <= a.starve) and not a.dead]
            """for animal in self.animals[val]:
                animal = [r for r in animal if r.eaten > animal.starve]
                
                self.animals[animal.kind].append(animal)
"""

    def reproduce(self):
        """ Rabbits reproduce like rabbits. """
        born = []
        for val in ITEM[1:]:
            for animal in self.animals[val]:
                for _ in range(rnd.randint(1, animal.max_offspring)):
                    born.append(animal.reproduce())

            # Capture field state for historical tracking
            #self.history[animal].append(self.num_animals())

            self.animal += born


    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_animals(self):
        all_animals = np.zeros(shape=(SIZE, SIZE), dtype=int)

        for val in ITEM[1:]:
            for r in self.animals[val]:
                all_animals[r.x, r.y] = 1
                return all_animals

    def num_animals(self):
        """ How many rabbits are there in the field ? """
        for val in ITEM[1:]:
            for animal in self.animals[val]:
                return len(animal)

    def amount_of_grass(self):
        return self.field.sum()

    def generation(self):
        """ Run one generation of rabbits """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()
"""
    def history(self, showTrack=True, showPercentage=True, marker='.'):

        plt.figure(figsize=(6, 6))
        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")

        xs = self.nrabbits[:]
        if showPercentage:
            maxrabbit = max(xs)
            xs = [x / maxrabbit for x in xs]
            plt.xlabel("% Rabbits")

        ys = self.ngrass[:]
        if showPercentage:
            maxgrass = max(ys)
            ys = [y / maxgrass for y in ys]
            plt.ylabel("% Rabbits")

        if showTrack:
            plt.plot(xs, ys, marker=marker)
        else:
            plt.scatter(xs, ys, marker=marker)

        plt.grid()

        plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history.png", bbox_inches='tight')
        plt.show()

    def history2(self):
        xs = self.nrabbits[:]
        ys = self.ngrass[:]

        sns.set_style('dark')
        f, ax = plt.subplots(figsize=(7, 6))

        sns.scatterplot(x=xs, y=ys, s=5, color=".15")
        sns.histplot(x=xs, y=ys, bins=50, pthresh=.1, cmap="mako")
        sns.kdeplot(x=xs, y=ys, levels=5, color="r", linewidths=1)
        plt.grid()
        plt.xlim(0, max(xs) * 1.2)

        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")
        plt.title("Rabbits vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.savefig("history2.png", bbox_inches='tight')
        plt.show()
"""

def animate(i, field, im):
    field.generation()
    # print("AFTER: ", i, np.sum(field.field), len(field.rabbits))
    im.set_array(field.field)
    plt.title("generation = " + str(i))
    return im,


def main():
    # Create the ecosystem
    field = Field()

    # create rabbits
    for _ in range(50):
        field.add_animal(Animal(2, max_offspring=2, speed=1, starve=0, eats=(1,)))

    # create foxes
    for _ in range(50):
        field.add_animal(Animal(3, max_offspring=1, speed=2, starve=10, eats=(2,)))



    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(array, cmap='PiYG', interpolation='hamming', aspect='auto', vmin=0, vmax=1)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=1000000, interval=1, repeat=True)
    plt.show()

    #field.history()
    #field.history2()


if __name__ == '__main__':
    main()

"""
def main():

    # Create the ecosystem
    field = Field()
    for _ in range(10):
        field.add_rabbit(Rabbit())


    # Run the world
    gen = 0

    while gen < 500:
        field.display(gen)
        if gen % 100 == 0:
            print(gen, field.num_rabbits(), field.amount_of_grass())
        field.move()
        field.eat()
        field.survive()
        field.reproduce()
        field.grow()
        gen += 1

    plt.show()
    field.plot()

if __name__ == '__main__':
    main()


"""




