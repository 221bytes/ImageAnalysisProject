package main

import (
	"math/rand"
	"time"

	"fmt"
)

const probCrossover = 1
const probMutation = 0.05
const iterationLimit = 1000
const populationSize = 100
const maxOne = 1024

type Genome [maxOne]int
type Children [2]Genome
type Fitness struct {
	fitnessScore int
	genome       Genome
}

type Population [populationSize]Genome
type FitnessPopulation map[int]Fitness

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func initialPopulation() *Population {
	data := Population{}
	for i := range data {
		for j := range data[i] {
			data[i][j] = rand.Int() % 2
		}
	}
	return &data
}

func printPop(pop Population) {
	for i := range pop {
		fmt.Println(pop[i])
	}
}

func printFitness(f *FitnessPopulation) {
	for i := range *f {
		fmt.Println((*f)[i])
	}
}

func fitness(pop *Population) *FitnessPopulation {
	fitnessPopulation := &FitnessPopulation{}
	for i := range pop {
		sum := 0
		genome := pop[i]
		for j := range genome {
			sum += genome[j]
		}
		(*fitnessPopulation)[i] = Fitness{fitnessScore: sum, genome: genome}
	}
	return fitnessPopulation
}

func checkStop(fitPop *FitnessPopulation, iteration int) bool {
	bestFitness := 0
	f := *fitPop
	for i := range f {
		if f[bestFitness].fitnessScore < f[i].fitnessScore {
			bestFitness = i
		}
		if f[i].fitnessScore == maxOne {
			fmt.Println("done", iteration, f[bestFitness])
			return true
		}
	}
	return false
}

func getRandomInter(min, max int) int {
	return rand.Intn(max-min) + min
}

func selection(fitPop *FitnessPopulation) *Population {
	parentsPop := &Population{}
	f := *fitPop
	for index := range f {
		randNB0 := getRandomInter(0, populationSize)
		randNB1 := getRandomInter(0, populationSize)
		if f[randNB0].fitnessScore > f[randNB1].fitnessScore {
			parentsPop[index] = f[randNB0].genome
		} else {
			parentsPop[index] = f[randNB1].genome
		}
	}
	return parentsPop
}

func crossover(children *Children) {
	father, mother := children[0], children[1]
	i := getRandomInter(0, maxOne)
	j := getRandomInter(i, maxOne)

	kid0, kid1 := father, mother
	for i <= j {
		kid0[i] = mother[i]
		kid1[i] = father[i]
		i++
	}
	children[0] = kid0
	children[1] = kid1
}

func mutation(child *Genome) {
	randNB := getRandomInter(0, maxOne)
	if child[randNB] == 0 {
		child[randNB] = 1
	} else {
		child[randNB] = 0
	}
}

func breedPopulation(fitPop *FitnessPopulation) *Population {
	parents := selection(fitPop)
	nextPopulation := &Population{}
	for father := 0; father < populationSize; father += 2 {
		if father+1 == populationSize {
			nextPopulation[father] = parents[father]
			break
		}
		mother := father + 1
		children := Children{parents[father], parents[mother]}
		if rand.Float64() < probCrossover == true {
			crossover(&children)
		}
		if rand.Float64() < probMutation == true {
			mutation(&children[0])
		}
		if rand.Float64() < probMutation == true {
			mutation(&children[1])
		}
		nextPopulation[father] = children[0]
		nextPopulation[mother] = children[1]
	}
	return nextPopulation
}

func main() {
	pop := initialPopulation()
	start := time.Now()
	for iteration := 0; iteration < iterationLimit; iteration++ {
		f := fitness(pop)
		if checkStop(f, iteration) {
			break
		}
		pop = breedPopulation(f)
	}
	elapsed := time.Since(start)
	fmt.Println(elapsed)
}
