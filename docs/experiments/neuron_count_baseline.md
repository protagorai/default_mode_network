# Neural Complexity and Emergent Behavior Baseline

**Source:** [List of animals by number of neurons - Wikipedia](https://en.wikipedia.org/wiki/List_of_animals_by_number_of_neurons)

## Overview

This document establishes a baseline for neural complexity across biological systems to inform experiments on emergent behaviors in artificial neural networks. The data presented here will guide the design of experiments aimed at demonstrating self-awareness and self-preservation behaviors in sufficiently complex neural networks.

Neuron counts constitute an important source of insight on the topic of neuroscience and intelligence: the question of how the evolution of a set of components and parameters (~10¹¹ neurons, ~10¹⁴ synapses) of a complex system leads to the phenomenon of intelligence. The human brain contains **86 billion neurons** with **16 billion neurons in the cerebral cortex**, representing the pinnacle of known neural complexity and self-awareness.

## Experimental Context

Understanding the relationship between neural complexity and emergent behaviors is fundamental to designing meaningful experiments. This framework will be used to:

1. **Establish complexity thresholds** for emergent self-awareness behaviors
2. **Design network topologies** that mirror biological neural organization
3. **Create benchmark simulations** that test for self-preservation instincts
4. **Validate emergent properties** against known biological baselines

## Complete Species Neural Complexity Table

| Species | Total Neurons | Forebrain/Cortex Neurons | Notes |
|---------|---------------|--------------------------|-------|
| [Sponge](https://en.wikipedia.org/wiki/Sponge) | 0 | — | No nervous system |
| [*Trichoplax*](https://en.wikipedia.org/wiki/Trichoplax) | 0 | — | No nervous system, exhibits coordinated behaviors |
| [*Asplanchna brightwellii*](https://en.wikipedia.org/wiki/Asplanchna_brightwellii) (rotifer) | ~200 | — | Brain only |
| [Tardigrade](https://en.wikipedia.org/wiki/Tardigrade) | ~200 | — | Brain only |
| [*Ciona intestinalis* larva](https://en.wikipedia.org/wiki/Ciona_intestinalis) (sea squirt) | 231 | — | Central nervous system only |
| [*Caenorhabditis elegans*](https://en.wikipedia.org/wiki/Caenorhabditis_elegans) (roundworm) | 302 | — | Complete connectome mapped |
| [Starfish](https://en.wikipedia.org/wiki/Starfish) | ~500 | — | Ring of neurons around mouth |
| [*Hydra vulgaris*](https://en.wikipedia.org/wiki/Hydra_vulgaris) | 5,600 | — | Fresh-water polyp |
| [*Megaphragma mymaripenne*](https://en.wikipedia.org/wiki/Megaphragma_mymaripenne) | 7,400 | — | Parasitic wasp |
| [Box jellyfish](https://en.wikipedia.org/wiki/Box_jellyfish) | 8,700–17,500 | — | *Tripedalia cystophora*, excludes rhopalia neurons |
| [Medicinal leech](https://en.wikipedia.org/wiki/Leech) | 10,000 | — | |
| [Pond snail](https://en.wikipedia.org/wiki/Lymnaeidae) | 11,000 | — | |
| [Sea slug](https://en.wikipedia.org/wiki/California_sea_slug) | 18,000 | — | *Aplysia* |
| [Amphioxus](https://en.wikipedia.org/wiki/Amphioxus) | 20,000 | — | Central nervous system only |
| [Larval zebrafish](https://en.wikipedia.org/wiki/Zebrafish) | 100,000 | — | |
| [Fruit fly](https://en.wikipedia.org/wiki/Drosophila) | 150,000 | 2,500 | *Drosophila melanogaster*, connectome mapped |
| [Wandering spider](https://en.wikipedia.org/wiki/Wandering_spider) | 100,000 | — | |
| [*Calliopsis*](https://en.wikipedia.org/wiki/Calliopsis) (bee) | 234,000 | — | |
| [Ant](https://en.wikipedia.org/wiki/Ant) | 250,000 | — | Varies per species |
| [*Perdita*](https://en.wikipedia.org/wiki/Perdita) (bee) | 275,000 | — | |
| [*Melissodes*](https://en.wikipedia.org/wiki/Melissodes) | 495,000 | — | |
| [*Bombus impatiens*](https://en.wikipedia.org/wiki/Bombus_impatiens) | 557,000 | — | |
| [Western honey bee](https://en.wikipedia.org/wiki/Western_honey_bee) | 613,000 | 170,000 | |
| [Honey bee](https://en.wikipedia.org/wiki/Honey_bee) | 960,000 | — | |
| [Cockroach](https://en.wikipedia.org/wiki/Cockroach) | 1,000,000 | 200,000 | |
| [Coconut crab](https://en.wikipedia.org/wiki/Coconut_crab) | >1,000,000 | — | Million interneurons for olfaction |
| [California carpenter bee](https://en.wikipedia.org/wiki/California_carpenter_bee) | 1,180,000 | — | |
| [Steudner's dwarf gecko](https://en.wikipedia.org/wiki/Steudner%27s_dwarf_gecko) | 1,771,000 | — | |
| [Brown anole](https://en.wikipedia.org/wiki/Brown_anole) | 2,792,000 | — | |
| [*Mochlus sundevallii*](https://en.wikipedia.org/wiki/Mochlus_sundevallii) | 3,049,000 | — | |
| [Peloponnese slowworm](https://en.wikipedia.org/wiki/Peloponnese_slowworm) | 3,713,000 | — | |
| [Common house gecko](https://en.wikipedia.org/wiki/Common_house_gecko) | 3,988,000 | — | |
| [*Takydromus sexlineatus*](https://en.wikipedia.org/wiki/Takydromus_sexlineatus) | 4,021,000 | — | |
| [*Anolis cristatellus*](https://en.wikipedia.org/wiki/Anolis_cristatellus) | 4,270,000 | — | |
| [Papua snake lizard](https://en.wikipedia.org/wiki/Papua_snake_lizard) | 4,271,000 | — | |
| [Guppy](https://en.wikipedia.org/wiki/Guppy) | 4,300,000 | — | |
| [Frog](https://en.wikipedia.org/wiki/Frog) | 16,000,000 | — | |
| [Adult zebrafish](https://en.wikipedia.org/wiki/Zebrafish) | ~10,000,000 | — | |
| [Naked mole-rat](https://en.wikipedia.org/wiki/Naked_mole-rat) | 26,880,000 | 6,000,000 | |
| [Little free-tailed bat](https://en.wikipedia.org/wiki/Little_free-tailed_bat) | 35,000,000 | 6,000,000 | |
| [Smoky shrew](https://en.wikipedia.org/wiki/Smoky_shrew) | 36,000,000 | 10,000,000 | |
| [Short-tailed shrew](https://en.wikipedia.org/wiki/Short-tailed_shrew) | 52,000,000 | 12,000,000 | |
| [Hottentot golden mole](https://en.wikipedia.org/wiki/Hottentot_golden_mole) | 65,000,000 | 22,000,000 | |
| [House mouse](https://en.wikipedia.org/wiki/House_mouse) | 71,000,000 | 14,000,000 | *Mus musculus* |
| [Golden hamster](https://en.wikipedia.org/wiki/Golden_hamster) | 90,000,000 | 17,000,000 | |
| [Star-nosed mole](https://en.wikipedia.org/wiki/Star-nosed_mole) | 131,000,000 | 17,000,000 | |
| [Zebra finch](https://en.wikipedia.org/wiki/Zebra_finch) | 131,000,000 | 55,000,000 | Brain only |
| [Eurasian blackcap](https://en.wikipedia.org/wiki/Eurasian_blackcap) | 157,000,000 | 52,000,000 | |
| [Goldcrest](https://en.wikipedia.org/wiki/Goldcrest) | 164,000,000 | 64,000,000 | |
| [Brown rat](https://en.wikipedia.org/wiki/Brown_rat) | 200,000,000 | 31,000,000 | *Rattus norvegicus* |
| [Red junglefowl](https://en.wikipedia.org/wiki/Red_junglefowl) | 221,000,000 | 61,000,000 | |
| [Great tit](https://en.wikipedia.org/wiki/Great_tit) | 226,000,000 | 83,000,000 | |
| [Guinea pig](https://en.wikipedia.org/wiki/Guinea_pig) | 240,000,000 | 43,510,000 | |
| [Gray mouse lemur](https://en.wikipedia.org/wiki/Gray_mouse_lemur) | 254,710,000 | 22,310,000 | |
| [Common treeshrew](https://en.wikipedia.org/wiki/Common_treeshrew) | 261,000,000 | 60,000,000 | |
| [Pigeon](https://en.wikipedia.org/wiki/Pigeon) | 310,000,000 | 72,000,000 | Brain only |
| [Budgerigar](https://en.wikipedia.org/wiki/Budgerigar) | 322,000,000 | 149,000,000 | |
| [Common blackbird](https://en.wikipedia.org/wiki/Common_blackbird) | 379,000,000 | 136,000,000 | |
| [Ferret](https://en.wikipedia.org/wiki/Ferret) | 404,000,000 | 38,950,000 | |
| [Cockatiel](https://en.wikipedia.org/wiki/Cockatiel) | 453,000,000 | 258,000,000 | |
| [Gray squirrel](https://en.wikipedia.org/wiki/Gray_squirrel) | 453,660,000 | 77,330,000 | |
| [Banded mongoose](https://en.wikipedia.org/wiki/Banded_mongoose) | 454,000,000 | 115,770,000 | |
| [Prairie dog](https://en.wikipedia.org/wiki/Prairie_dog) | 473,940,000 | 53,770,000 | |
| [Common starling](https://en.wikipedia.org/wiki/Common_starling) | 483,000,000 | 226,000,000 | |
| [European rabbit](https://en.wikipedia.org/wiki/European_rabbit) | 494,200,000 | 71,450,000 | |
| [Octopus](https://en.wikipedia.org/wiki/Octopus) | 500,000,000 | — | |
| [Bigfin reef squid](https://en.wikipedia.org/wiki/Bigfin_reef_squid) | >500,000,000 | — | |
| [Common marmoset](https://en.wikipedia.org/wiki/Common_marmoset) | 636,000,000 | 245,000,000 | |
| [Eastern rosella](https://en.wikipedia.org/wiki/Eastern_rosella) | 642,000,000 | 333,000,000 | |
| [Barn owl](https://en.wikipedia.org/wiki/Barn_owl) | 690,000,000 | 437,000,000 | |
| [Monk parakeet](https://en.wikipedia.org/wiki/Monk_parakeet) | 697,000,000 | 396,000,000 | |
| [Azure-winged magpie](https://en.wikipedia.org/wiki/Azure-winged_magpie) | 741,000,000 | 400,000,000 | |
| [Rock hyrax](https://en.wikipedia.org/wiki/Rock_hyrax) | 756,000,000 | 198,000,000 | |
| [Cat](https://en.wikipedia.org/wiki/Cat) | 760,000,000 | 249,830,000 | |
| [Black-rumped agouti](https://en.wikipedia.org/wiki/Black-rumped_agouti) | 857,000,000 | 113,000,000 | |
| [Magpie](https://en.wikipedia.org/wiki/Magpie) | 897,000,000 | 443,000,000 | |
| [Common hill myna](https://en.wikipedia.org/wiki/Common_hill_myna) | 906,000,000 | 410,000,000 | |
| [Western jackdaw](https://en.wikipedia.org/wiki/Western_jackdaw) | 968,000,000 | 492,000,000 | |
| [Raccoon dog](https://en.wikipedia.org/wiki/Raccoon_dog) | 1,160,000,000 | 240,180,000 | |
| [Emu](https://en.wikipedia.org/wiki/Emu) | 1,335,000,000 | 439,000,000 | |
| [Three-striped night monkey](https://en.wikipedia.org/wiki/Three-striped_night_monkey) | 1,468,000,000 | 442,000,000 | |
| [Rook](https://en.wikipedia.org/wiki/Rook_(bird)) | 1,509,000,000 | 820,000,000 | |
| [Grey parrot](https://en.wikipedia.org/wiki/Grey_parrot) | 1,566,000,000 | 850,000,000 | |
| [Capybara](https://en.wikipedia.org/wiki/Capybara) | 1,600,000,000 | 306,500,000 | |
| [Common ostrich](https://en.wikipedia.org/wiki/Common_ostrich) | 1,620,000,000 | 479,410,000 | |
| [Jackal](https://en.wikipedia.org/wiki/Jackal) | 1,730,000,000 | 393,620,000 | |
| [Fox](https://en.wikipedia.org/wiki/Fox) | 2,110,000,000 | 355,010,000 | |
| [Sulphur-crested cockatoo](https://en.wikipedia.org/wiki/Sulphur-crested_cockatoo) | 2,122,000,000 | 1,135,000,000 | |
| [Raccoon](https://en.wikipedia.org/wiki/Raccoon) | 2,148,000,000 | 453,000,000 | |
| [Kea](https://en.wikipedia.org/wiki/Kea) | 2,149,000,000 | 1,281,000,000 | |
| [Raven](https://en.wikipedia.org/wiki/Raven) | 2,171,000,000 | 1,204,000,000 | Brain only |
| [Domestic pig](https://en.wikipedia.org/wiki/Domestic_pig) | 2,220,000,000 | 425,000,000 | |
| [Dog](https://en.wikipedia.org/wiki/Dog) | 2,253,000,000 | 627,000,000 | Average across breeds |
| [Blue-and-yellow macaw](https://en.wikipedia.org/wiki/Blue-and-yellow_macaw) | 3,136,000,000 | 1,900,000,000 | Brain only |
| [Common squirrel monkey](https://en.wikipedia.org/wiki/Common_squirrel_monkey) | 3,246,000,000 | 1,340,000,000 | |
| [Crab-eating macaque](https://en.wikipedia.org/wiki/Crab-eating_macaque) | 3,440,000,000 | 800,960,000 | |
| [Tufted capuchin](https://en.wikipedia.org/wiki/Tufted_capuchin) | 3,691,000,000 | 1,140,000,000 | |
| [Bonnet macaque](https://en.wikipedia.org/wiki/Bonnet_macaque) | 3,780,000,000 | 1,660,000,000 | |
| [Striped hyena](https://en.wikipedia.org/wiki/Striped_hyena) | 3,885,000,000 | 495,280,000 | |
| [Lion](https://en.wikipedia.org/wiki/Lion) | 4,667,000,000 | 545,240,000 | |
| [Rhesus macaque](https://en.wikipedia.org/wiki/Rhesus_macaque) | 6,376,000,000 | 1,710,000,000 | |
| [Brown bear](https://en.wikipedia.org/wiki/Brown_bear) | 9,586,000,000 | 250,970,000 | |
| [Giraffe](https://en.wikipedia.org/wiki/Giraffe) | 10,750,000,000 | 1,731,000,000 | |
| [Yellow baboon](https://en.wikipedia.org/wiki/Yellow_baboon) | 10,950,000,000 | 2,880,000,000 | |
| [Chimpanzee](https://en.wikipedia.org/wiki/Chimpanzee) | 28,000,000,000 | 7,400,000,000 | |
| [Orangutan](https://en.wikipedia.org/wiki/Orangutan) | 32,600,000,000 | 7,704,000,000–8,900,000,000 | |
| [Gorilla](https://en.wikipedia.org/wiki/Gorilla) | 33,400,000,000 | 9,100,000,000 | |
| [Common minke whale](https://en.wikipedia.org/wiki/Common_minke_whale) | 57,000,000,000 | 3,134,000,000–12,800,000,000 | |
| [Human](https://en.wikipedia.org/wiki/Human) | 86,000,000,000 | 16,340,000,000 | Average adult |
| [Short-finned pilot whale](https://en.wikipedia.org/wiki/Short-finned_pilot_whale) | 128,000,000,000 | 11,850,000,000 | |
| [African elephant](https://en.wikipedia.org/wiki/African_elephant) | 257,000,000,000 | 5,600,000,000 | |

## Key Insights for Experimental Design

### Complexity Thresholds
- **Basic coordination**: Observable in organisms with 0 neurons (*Trichoplax*)
- **Simple behaviors**: 200-20,000 neurons (rotifer to amphioxus level)
- **Complex behaviors**: 150,000+ neurons (fruit fly level with mapped connectome)
- **Social behaviors**: 960,000 neurons (honeybee level)
- **Problem solving**: 500,000,000+ neurons (cephalopod level)  
- **Self-awareness**: 86,000,000,000+ neurons (human level)

### Critical Observations
1. **C. elegans** with only 302 neurons demonstrates the importance of connectivity (complete connectome available)
2. **Connectome mapping**: Both C. elegans (302 neurons) and fruit fly (150,000 neurons) have complete neural wiring diagrams
3. **Intelligence vs. total neurons**: Elephants (257B total) have more neurons than humans (86B) but fewer cortical neurons (5.6B vs 16B)
4. **Cephalopod intelligence**: Octopi with 500M neurons show remarkable problem-solving abilities without centralized brain architecture
5. **Avian intelligence**: Ravens (2.2B total, 1.2B pallium) and parrots exhibit complex cognition with high pallium neuron density
6. **Forebrain/cortex neurons**: Appears more predictive of cognitive abilities than total neuron count across species

## Experimental Implications

### Network Architecture Design
- **Minimal complexity**: Start with C. elegans-inspired 302-neuron networks for basic behavioral studies
- **Intermediate targets**: Fruit fly level (150K neurons) for complex behavior emergence
- **Social intelligence**: Honeybee level (960K neurons) for swarm and communication behaviors
- **Advanced cognition**: Mammalian cortical neuron counts (millions to billions) for self-awareness experiments
- **Focus on connectivity**: Emphasize network topology and connection density over raw neuron count

### Behavioral Benchmarks
- **Basic reflexes**: 302-neuron baseline (C. elegans level)
- **Environmental navigation**: 150,000-neuron target (fruit fly level)
- **Social coordination**: 960,000-neuron threshold (honeybee level)
- **Tool use and problem solving**: 500,000,000-neuron complexity (octopus level)
- **Language and self-awareness**: 16,000,000,000+ cortical neurons (human level)

### Simulation Objectives
1. **Demonstrate emergent navigation** in simple 302-neuron networks
2. **Show complex environmental responses** at 150K-neuron scales  
3. **Achieve social coordination behaviors** at million-neuron scales
4. **Develop problem-solving capabilities** at 500M-neuron complexity
5. **Target self-awareness indicators** at billion+ neuron scales
6. **Validate against biological connectomes** where available (C. elegans, Drosophila)

## Methodological Notes

### Measurement Methods
- **Isotropic fractionator**: Most reliable method for neuron counting (used for most mammals)
- **Optical fractionator**: Traditional method, tends to overestimate (marked with * in source)
- **Estimated values**: Based on brain mass correlations (marked with ^ in source)
- **Connectome mapping**: Complete neural wiring diagrams available for C. elegans and Drosophila

### Intelligence Predictors
The current best predictor for animal intelligence is the number of neurons in the forebrain:
- **Mammals**: Cerebral cortex (pallium) neuron count
- **Birds**: Dorsal ventricular ridge (DVR) of the pallium
- **Insects**: Corpora pedunculata (mushroom bodies)

This accounts for variation in other brain regions (like cerebellum) that show no established link to intelligence.

## References

- Herculano-Houzel, S. (2009). The human brain in numbers: a linearly scaled-up primate brain. *Frontiers in Human Neuroscience*, 3, 31.
- White, J. G., et al. (1986). The structure of the nervous system of the nematode Caenorhabditis elegans. *Philosophical Transactions of the Royal Society of London*, 314(1165), 1-340.
- Azevedo, F. A., et al. (2009). Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled‐up primate brain. *Journal of Comparative Neurology*, 513(5), 532-541.
