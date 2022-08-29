## Chiller efficiency inputs
Chiller efficiency depends on 4 inputs:
* Water to refrigerant heat transfer
* Refrigerant cycle thermodynamics
* Power conversion and transmission efficiency
* Centrifugal compressor efficiency

These are broad categories, and most of each category cannot be changed once a piece of equipment is purchased. For example, power conversion efficiency and compressor efficiency cannot be improved once a chiller is bought without changing mechanical components. There are few things that an owner can change about how they operate equipment once it is purchased, and this blog will describe the owners strategies for optimally operating chillers.

## How can someone improve water to refrigerant heat transfer?
* Condenser water temperature

There are two general spots where heat transfer happens in a chiller: the condenser, and evaporator. The condenser transfers heat from the refrigerant to condenser medium (water, in the case of a water cooled chiller), and from the evaporative medium (water, in a chilled water system) to refrigerant. There is a lot of research that goes into efficiently transferring heat between mediums, but this project isn't about any of that. When the condenser water is cold, it is easier to transfer heat compared to hot condenser water. Therefore, condenser water temperature has a large influence on the efficiency of heat transfer, and the efficiency of the chiller. Condenser water temperature will be used as a predictor of chiller efficiency to predict optimal operating points.

Note: It is possible to further optimize the total efficiency of a chiller plant by using more fan energy to reduce condenser water temperature, which will save energy at the chiller. While interesting, this project does not attempt to predict the most energy efficient tradeoff between condenser fan speed and chiller capacity.

* Rated cooling capacity, and desired cooling capacity

Some equipment has the capability to vary capacity from minimum to rated maximum.  Chillers operate at different efficiencies when output load is varied up to its maximum capacity. Generally, chillers are least efficient when at minimum rated capacity, and most efficient somewhere less than full capacity. This implies there is a global maximum efficiency that is neither at the minimum or maximum capacity.

* Return and supply evaporator water temperature

Higher return evaporator water temperature can have a positive effect on water to refrigerant heat transfer, and possibly increase efficiency. Generally, higher supply water temperature (not too high!) also increases heat transfer efficiency. 

## How can someone choose to operate at the most efficient capacity?
By measuring cooling capacity output and electric power input across all operating inputs it is possible to know where equipment operates most efficiently. Once the most efficient operating point is known, and given a set of independent variable inputs, it is possible to choose the number of chillers, and output capacity of each chiller, which produces equal to or greater than the desired cooling capacity while minimizing electric power consumed.

Independent variables:
* Cooling load (power) required [kW]
* Condenser water temperature [DEG C]
* Condenser water flow rate
* Number of operable chillers [integer]
* Evaporator return water temperature
* Evaporator supply water temperature (setpoint)
* Evaporator water flow rate

Output:
* Number of chillers to enable
* Operating capacity of each chiller