"""
Heidi Eren, Conor Doyle, Olivia Mintz, Kelsey Nihezagirwe
DS3500 HW 5
Evolution Animation
"""

import random as rnd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
import seaborn as sns
from collections import defaultdict
import matplotlib.colors as colors
import argparse
import pprint as pp

parser = argparse.ArgumentParser(description='Simulate a rabbit and fox population over time.')
parser.add_argument('--grass_growth_rate', type=float, default=0.8, help='the rate at which grass grows each turn')
parser.add_argument('--fox_k_value', type=float, default=30, help='the K value for foxes')
parser.add_argument('--field_size', type=int, default=50, help='the size of the field')
parser.add_argument('--initial_rabbits', type=int, default=2, help='the number of rabbits at the start of the simulation')
parser.add_argument('--initial_foxes', type=int, default=5, help='the number of foxes at the start of the simulation')

args = parser.parse_args()

SIZE = args.field_size  # The dimensions of the field
R_OFFSPRING = 2  # Max offspring when a rabbit reproduces
GRASS_RATE = args.grass_growth_rate  # Probability that grass grows back at any location in the next season.
WRAP = False  # Does the field wrap around on itself when rabbits move?
ITEM = [0,1,2,3]
TYPE = ['empty', 'grass', 'rabbit', 'fox']
SPEED = 1

WHOLE_FIELD = []

# parameters
# max_offspring, speed, how long they go without eating, what the animal can eat
# give each food
# to differentiate between animals == what it eats, choose a number to corresponding to what it eats
# the number is a value in the init
# frames is the k
# if it equals this number, change this


class Animal:
    """ Rabbits and foxes in a field of grass """

    def __init__(self, val, max_offspring, speed, starve, eats):
        self.val = val
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

        if amount == 0:
            self.last_eaten += 1
        else:
            self.last_eaten = 0


        # keep track of how many times animal last ate


    def move(self):
        """ Move up, down, left, right randomly """

        if WRAP:
            self.x = (self.x + rnd.choice([-self.speed, self.speed])) % SIZE
            self.y = (self.y + rnd.choice([-self.speed, self.speed])) % SIZE

        else:
            self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-self.speed, self.speed]))))
            self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-self.speed, self.speed]))))


    def __repr__(self):
        return(TYPE[self.val])


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
        self.animals[animal.val].append(animal)


    def move(self):
        """ Move the animal """
        for val in ITEM[2:]:
            for r in self.animals[val]:
                r.move()


    def update_location_field(self):
        """Updates location of field"""
        self.coordinates.clear()
        for val in ITEM[2:]:
            for animal in self.animals[val]:
                self.coordinates[(animal.x, animal.y)].append(animal)

        for val in ITEM[2:]:
            self.location[val] = self.get_animals(val)


    def eat(self):
        """ Rabbits eat grass (if they find grass where they are),
        foxes eat rabbits (if they find rabbits where they are) """


        self.update_location_field()

        for val in ITEM[2:]:
            for animal in self.animals[val]:
                for prey in animal.eats:
                    if prey == 1:
                        animal.eat(self.field[animal.x, animal.y])
                        self.field[animal.x, animal.y] = 0
                    else:
                        # for every other animal at the same location, check whether it can be eaten
                        ate = False
                        for other in self.coordinates[animal.x, animal.y]:
                            if other.val in animal.eats:
                                animal.eat(1)
                                ate = True
                                other.dead = True
                        if not ate:
                            animal.eat(0)


        """
        # account for rabbits eating habits
        for animal in self.animals[2]:

            # check if field has grass
            if self.field[animal.x, animal.y] == animal.eats[0]:

                animal.eat(self.field[animal.x, animal.y])
                print('rabbits are munching')
                self.field[animal.x, animal.y] = 0
                print("field", self.field[animal.x, animal.y])

            else:
                # rabbit has nothing to eat
                animal.eat(0)
                animal.dead = True

    # account for foxes eating habits
            for fox in self.animals[3]:
                # check if fox is on a rabbit
                if (fox.x == animal.x) and (fox.y == animal.y):
                    fox.eat(1)
                    animal.dead = True
                else:
                    fox.eat(0)"""


    def survive(self):
        """ Rabbits who eat some grass live to eat another day """

        for val in ITEM[2:]:
            #for a in self.animals[val]:
                #print(f"last eaten{a.last_eaten}")
            self.animals[val] = [a for a in self.animals[val] if (a.last_eaten <= a.starve) and not a.dead]


    def reproduce(self):
        """ Rabbits reproduce like rabbits. """


        for val in ITEM[2:]:
            born = []
            for animal in self.animals[val]:

                #if animal.last_eaten <= animal.starve:
                    # check if animal has eaten in the current cycle
                if animal.eaten > 0:
                    for _ in range(rnd.randint(1, animal.max_offspring)):
                        born.append(animal.reproduce())

            self.animals[val] += born
            #born = []  reset the list for the next animal type
            print(val,":", len(self.animals[val]))
            # Capture field state for historical tracking

        # update the amount of grass in the field
        self.history[1].append(self.amount_of_grass())

        # update the number of animals in the field
        for val in ITEM[2:]:
            self.history[val].append(self.num_animals(val))

            # Capture field state for historical tracking
            #self.nrabbits.append(self.num_rabbits())
            #self.ngrass.append(self.amount_of_grass())
    def grow(self):
        """ Grass grows back with some probability """
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def get_animals(self, val):

        all_animals = np.zeros(shape=(SIZE, SIZE), dtype=int)

        for animal in self.animals[val]:
            all_animals[animal.x, animal.y] = val

        return all_animals

        """all_animals = self.field
        #all_animals = np.ones(shape=(SIZE, SIZE), dtype=int)

        #print(all_animals)

        #pp.pprint(self.animals)

        for val in ITEM[2:]:
            for r in self.animals[val]:

                all_animals[r.x, r.y] = val


        return all_animals"""

    def num_animals(self, val):
        """ How many rabbits are there in the field ? """
        return len(self.animals[val])

    def amount_of_grass(self):
        return self.field.sum()

    def generation(self, speed):
        """ Run one generation of rabbits """
        #print(all_animals)
        #print("before we move")

        for s in range(speed):
            self.move()
            self.eat()
            self.survive()
            self.reproduce()
            self.grow()

    def history_graph(self, showTrack=True, showPercentage=True, marker='.'):

        rabbit_count = []
        fox_count = []
        grass_count = []
        empty_count = []

        for k in range(len(self.history)):
            rabbit_count.append(self.history[k][2])
            fox_count.append(self.history[k][3])
            grass_count.append(self.history[k][1])
            empty_count.append(SIZE ** 2 - sum(self.history[k][1:]) - rabbit_count[k] - fox_count[k])

        plt.plot(rabbit_count, label='Rabbits')
        plt.plot(fox_count, label='Foxes')
        plt.plot(grass_count, label='Grass')
        plt.plot(empty_count, label='Empty')

        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Number of animals/spaces')
        plt.title('Population dynamics over time')
        plt.show()


        plt.figure(figsize=(6, 6))
        plt.xlabel("generation #")
        plt.ylabel("% population")
        for val in ITEM[2:]:
            for animal in self.history[val]:
                xs = animal[:]
                if showPercentage:
                    max_animal = max(xs)
                    xs = [x / max_animal for x in xs]
                    plt.xlabel("generation #")

        ys = self.history[1]
        if showPercentage:
            maxgrass = max(ys)
            ys = [y / maxgrass for y in ys]
            plt.ylabel("% population")

        if showTrack:
            plt.plot(xs, ys, marker=marker)
        else:
            plt.scatter(xs, ys, marker=marker)
        plt.grid()
        plt.title("Animals vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.legend(TYPE)
        plt.savefig("history.png", bbox_inches='tight')
        plt.show()
    def history1(self, showTrack=True, showPercentage=True, marker='.'):


        nrabbits = self.history[2]
        nfoxes = self.history[3]
        ngrass = self.history[1]

        plt.figure(figsize=(6, 6))
        plt.xlabel("# Rabbits")
        plt.ylabel("# Grass")

        xs = nrabbits[:]
        if showPercentage:
            maxrabbit = max(xs)
            xs = [x / maxrabbit for x in xs]
            plt.xlabel("% Rabbits")

        ys = ngrass[:]
        if showPercentage:
            maxgrass = max(ys)
            ys = [y / maxgrass for y in ys]
            plt.ylabel("% Rabbits")

        # Add data for foxes
        zs = nfoxes[:]
        if showPercentage:
            maxfox = max(zs)
            zs = [z / maxfox for z in zs]
            plt.ylabel("% Foxes")

        if showTrack:
            plt.plot(xs, ys, marker=marker, label='Rabbits')
            plt.plot(xs, zs, marker=marker, label='Foxes')
        else:
            plt.scatter(xs, ys, marker=marker, label='Rabbits')
            plt.scatter(xs, zs, marker=marker, label='Foxes')

        plt.grid()

        plt.title("Rabbits vs. Foxes: GROW_RATE =" + str(GRASS_RATE))
        plt.legend()
        plt.savefig("history.png", bbox_inches='tight')
        plt.show()


    def history2(self):

        xs = self.history[2]
        ys = self.history[1]
        zs = self.history[3]

        sns.set_style('dark')
        f, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(x=xs, y=ys, s=5, color=".15", label="Grass")
        sns.histplot(x=xs, y=ys, bins=50, pthresh=.1, cmap="mako")
        sns.kdeplot(x=xs, y=ys, levels=5, color="r", linewidths=1, label="Rabbits")
        sns.scatterplot(x=xs, y=zs, s=5, color="b", label="Foxes")
        plt.grid()
        plt.xlim(0, max(xs) * 1.2)
        plt.xlabel("# Rabbits/Foxes")
        plt.ylabel("# Grass")
        plt.title("Rabbits and Foxes vs. Grass: GROW_RATE =" + str(GRASS_RATE))
        plt.legend(loc="upper left")
        plt.savefig("history2.png", bbox_inches='tight')
        plt.show()

    def plot_populations(self, showPercentage=True):
        """ Plots the population of rabbits, foxes, and grass over time """

        rabbits = self.history[2]
        grass = self.history[1]
        foxes = self.history[3]
        time = list(range(1,len(rabbits)+1))

        xs = rabbits[:]
        if showPercentage:
            maxrabbit = max(xs)
            xs = [x / maxrabbit for x in xs]
            # plot the lines
            plt.plot(time, xs, label='rabbit')

        ys = grass[:]
        if showPercentage:
            maxgrass = max(ys)
            ys = [x / maxgrass for x in ys]
            # plot the lines
            plt.plot(time, ys, label='grass')

        zs = foxes[:]
        if showPercentage:
            maxfoxes = max(zs)
            zs = [x / maxfoxes for x in zs]
            # plot the lines
            plt.plot(time, zs, label='foxes')


        # add title and labels for axes
        plt.title('Three Lines')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        # add legend and show the plot
        plt.legend()
        plt.show()

def animate(i, field, im):

    field.generation(SPEED)
    total_field = field.field

    for val in ITEM[2:]:
        total_field = np.maximum(total_field, field.location[val])

    # print("AFTER: ", i, np.sum(field.field), len(field.rabbits))
    #x = field.get_animals()
    #print(x)
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

    clist = ['tan','green', 'blue', 'red']
    my_cmap = colors.ListedColormap(clist)

    "plt.imshow(total, cmap=my_cmap, interpolation='none')"

    #pp.pprint(field.array)


    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(array, cmap=my_cmap, interpolation='None', aspect='auto', vmin=0, vmax=3)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im,), frames=1000000, interval=1, repeat=True)
    #pp.pprint(field.array)
    plt.show()

    pp.pprint(field.history)
    field.history1()

    #field.history2()

    field.plot_populations()


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