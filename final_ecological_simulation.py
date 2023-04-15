"""
Heidi Eren, Conor Doyle, Olivia Mintz, Kelsey Nihezagirwe
DS3500 HW 5
4/14/23
Evolution_animation.py

"""

import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
from collections import defaultdict
import matplotlib.colors as colors
import argparse

# create parsers to initialize command line arguments
parser = argparse.ArgumentParser(description='Simulate a rabbit and fox population over time.')
parser.add_argument('--grass_growth_rate', type=float, default=0.8, help='the rate at which grass grows each turn')
parser.add_argument('--fox_k_value', type=int, default=10, help='the number of cycles the fox can go without eating')
parser.add_argument('--field_size', type=int, default=50, help='the size of the field')
parser.add_argument('--initial_rabbits', type=int, default=5, help='the number of rabbits at the start of the simulation')
parser.add_argument('--initial_foxes', type=int, default=2, help='the number of foxes at the start of the simulation')

args = parser.parse_args()

# set global variables
SIZE = args.field_size  # The dimensions of the field
GRASS_RATE = args.grass_growth_rate  # Probability that grass grows back at any location in the next season.
WRAP = False  # The field wrap around on itself when the animals move
ITEM = [0,1,2,3] # Kind for each animal
TYPE = ['empty', 'grass', 'rabbit', 'fox']
SPEED = 1 # Set speed of generation animation


class Animal:
    """ Animal class that represents either a rabbit or fox on a field of grass

     Attributes:
         kind (int): the number assigned to the animal
         max_offspring (int): the maximum number of offspring an animal can have
         speed (int): the speed at which the animal is moving
         starve (int): the number of cycles an animal can last before dying
         eats (int): what animal they eat

         x (int): x coordinate of animal
         y (int): y coordinate of animal
         eaten (int): an amount that increases if the animal eats
         last_eaten (int): the number of generations since the last ate
         dead (bool): true if the animal is dead, false if the animal is alive"""

    def __init__(self, kind, max_offspring, speed, starve, eats):
        self.kind = kind
        self.max_offspring = max_offspring
        self.speed = speed
        self.starve = starve
        self.eats = eats

        # initialize location of animal
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)

        # set eating status of animal
        self.eaten = 0
        self.last_eaten = 0
        self.dead = False

    def reproduce(self):
        """ Reproduce the animal at the same location;
        each reproduced animal's eating level is reset to zero """

        self.eaten = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        """ Have the animal eat when possible
         amount (int): the value of whether the animal ate """

        # update eaten to the amount eaten
        self.eaten += amount

        # if animal didn't eat, update last_eaten
        if amount == 0:
            self.last_eaten += 1
        else:
            self.last_eaten = 0

    def move(self):
        """ Move up, down, left, right randomly """

        # if field wraps around itself
        if WRAP:
            self.x = (self.x + rnd.choice([-self.speed, self.speed])) % SIZE
            self.y = (self.y + rnd.choice([-self.speed, self.speed])) % SIZE

        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-self.speed, self.speed]))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-self.speed, self.speed]))))

    def __repr__(self):
        return(TYPE[self.kind])

class Field:
    """ A field is a patch of grass with 0 or more animals walking around

     Attributes:
         animals (dict): animal population where keys are animal kind, values are list of animals of the same kind
         field (array): an array of ones representing the field in th ecosystem
         history (dict): population of each generation, keys are the animal kind and number of animals of that kind
                        in that generation
         location (dict): stores the location of each animal
         coordinates (dict): the field's coordinates and what animal or plant is on that coordinate

     """

    def __init__(self):
        self.animals = defaultdict(list)
        self.field = np.ones(shape=(SIZE, SIZE), dtype=int)
        self.history = defaultdict(list)
        self.location = defaultdict(list)
        self.coordinates = defaultdict(list)

    def add_animal(self, animal):
        """ A new animal is added to the field
         animal (lst): user-inputted parameters associated with the animal """

        self.animals[animal.kind].append(animal)

    def move(self):
        """ Move the animal """
        for val in ITEM[2:]:
            for r in self.animals[val]:
                r.move()

    def update_location_field(self):
        """Updates the location of field for historical tracking """

        self.coordinates.clear()

        for val in ITEM[2:]:
            for animal in self.animals[val]:

                # record coordinates of animal into global variable
                self.coordinates[(animal.x, animal.y)].append(animal)

        for val in ITEM[2:]:

            # record location of animal
            self.location[val] = self.get_animals(val)

    def eat(self):
        """ Have the animal eat based on certain conditions;
        Rabbits eat grass (if they find grass where they are),
        foxes eat rabbits (if they find rabbits where they are) """

        self.update_location_field()

        for val in ITEM[2:]:
            for animal in self.animals[val]:

                # loop through what the animal eats
                for prey in animal.eats:

                    # check for grass-eaters, updates the animal's eat, and empty the field location
                    if prey == 1:
                        animal.eat(self.field[animal.x, animal.y])
                        self.field[animal.x, animal.y] = 0

                    else:
                        # animal did not eat
                        ate = False

                        # for every other animal at the same location, check whether it can be eaten
                        for other in self.coordinates[animal.x, animal.y]:
                            if other.kind in animal.eats:
                                animal.eat(1)
                                ate = True

                                # animal on the same coordinates dies
                                other.dead = True
                        # animal did not eat
                        if not ate:
                            animal.eat(0)

    def survive(self):
        """ Update the survival of the animals based on the number of cycles they can go without eating """

        for val in ITEM[2:]:
            self.animals[val] = [a for a in self.animals[val] if (a.last_eaten <= a.starve) and not a.dead]

    def reproduce(self):
        """ Reproduce the animal """

        for val in ITEM[2:]:
            born = []
            for animal in self.animals[val]:

                # check if the animal ate before reproducing
                if animal.eaten > 0:
                    for _ in range(rnd.randint(1, animal.max_offspring)):
                        born.append(animal.reproduce())

            self.animals[val] += born

        # update the amount of grass in the field
        self.history[1].append(self.amount_of_grass())

        for val in ITEM[2:]:

            # update the number of animals in the field
            self.history[val].append(self.num_animals(val))

    def grow(self):
        """ Grass grows back with some probability """
        grow_loc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, grow_loc)

    def get_animals(self, val):
        """ Plots the animals in the field
         val (int): kind of the animal """

        # empty pseudo field
        all_animals = np.zeros(shape=(SIZE, SIZE), dtype=int)

        for animal in self.animals[val]:
            # update the animal's location on the field
            all_animals[animal.x, animal.y] = val

        return all_animals

    def num_animals(self, val):
        """ number of animals in the field
        val (int): kind of the animal """

        return len(self.animals[val])

    def amount_of_grass(self):
        """ amount of grass in the field """

        sum_of_ones = np.sum(np.sum(self.field == 1))
        return sum_of_ones

    def generation(self, speed):
        """ Run one generation of animals
        speed (int): value that increases the speed of the generations (global variable)"""

        for s in range(speed):
            self.move()
            self.eat()
            self.survive()
            self.reproduce()
            self.grow()

    def history1(self, showTrack=True, showPercentage=True, marker='.'):
        """ Plots the history of the field
         showTrack (bool): Have graph track num of populations (display line graph)
         showPercentage (bool): Have graph display percent of animal population
         marker (str): marker on graph """

        # list of number of animals in each generation
        nrabbits = self.history[2]
        nfoxes = self.history[3]
        ngrass = self.history[1]

        # builds the plot
        plt.figure(figsize=(6, 6))
        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")

        # creates a copy of the rabbit population
        xs = nrabbits[:]

        # find the percentage of rabbit population in each cycle
        if showPercentage:
            maxrabbit = max(xs)
            xs = [x / maxrabbit for x in xs]
            plt.xlabel("% Rabbits")

        # grass population
        ys = ngrass[:]
        if showPercentage:
            maxgrass = max(ys)
            ys = [y / maxgrass for y in ys]
            plt.ylabel("% Rabbits")

        # fox population
        zs = nfoxes[:]
        if showPercentage:
            maxfox = max(zs)
            zs = [z / maxfox for z in zs]
            plt.ylabel("% Foxes")

        # sets type of graph
        if showTrack:
            plt.plot(xs, ys, marker=marker, label='Rabbits')
            plt.plot(xs, zs, marker=marker, label='Foxes')
        else:
            plt.scatter(xs, ys, marker=marker, label='Rabbits')
            plt.scatter(xs, zs, marker=marker, label='Foxes')

        plt.grid()

        # add labels to the graph
        plt.title("Rabbits vs. Foxes: GROW_RATE =" + str(GRASS_RATE))
        plt.legend()
        plt.savefig("Animal_growth_rate.png", bbox_inches='tight')
        plt.show()

    def history2(self, showPercentage=True):
        """ Plots the population of rabbits, foxes, and grass over time
         showPercentage (bool): Have graph display percent of animal population """

        # list of number of animals in each generation
        rabbits = self.history[2]
        grass = self.history[1]
        foxes = self.history[3]
        time = list(range(1,len(rabbits)+1))

        # rabbit population
        xs = rabbits[:]
        if showPercentage:
            maxrabbit = max(xs)
            xs = [x / maxrabbit for x in xs]
            plt.plot(time, xs, label='rabbit')

        # grass population
        ys = grass[:]
        if showPercentage:
            maxgrass = max(ys)
            ys = [x / maxgrass for x in ys]
            plt.plot(time, ys, label='grass')

        # fox population
        zs = foxes[:]
        if showPercentage:
            maxfoxes = max(zs)
            zs = [x / maxfoxes for x in zs]
            plt.plot(time, zs, label='foxes')

        # add title and labels for axes
        plt.title('Population of Rabbits, Foxes and Grass Over Time')
        plt.xlabel('Number of Generations')
        plt.ylabel('% Population')
        plt.legend()
        plt.savefig("Percent_change_in_population.png", bbox_inches='tight')
        plt.show()

def animate(i, field, im):
    """ Creates an animation of each cycle
    i (int): cycle number
    field (object): the ecosystem initalized
    im: outputs Animation of field """

    # set the speed of the generation
    field.generation(SPEED)

    total_field = field.field
    for val in ITEM[2:]:
        total_field = np.maximum(total_field, field.location[val])

    # set the field in the animation
    im.set_array(total_field)
    plt.title("generation = " + str(i * SPEED))
    return im,


def main():

    # Create the ecosystem
    field = Field()

    # create rabbits
    for _ in range(args.initial_rabbits):
        field.add_animal(Animal(2, max_offspring=2, speed=1, starve=0, eats=(1,)))

    # create foxes
    for _ in range(args.initial_foxes):
        field.add_animal(Animal(3, max_offspring=1, speed=2, starve=args.fox_k_value, eats=(2,)))

    # set colors for field
    clist = ['tan','green', 'blue', 'red']
    my_cmap = colors.ListedColormap(clist)

    # create animation
    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(array, cmap=my_cmap, interpolation='None', aspect='auto', vmin=0, vmax=3)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=1000000, interval=1, repeat=True)
    plt.show()

    # create graphs
    field.history1()
    field.history2()


if __name__ == '__main__':

    main()
