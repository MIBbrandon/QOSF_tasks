from typing import List

from qiskit import QuantumCircuit, execute, Aer
import qiskit.providers.aer.noise as noise
# noinspection PyPep8Naming
from random import uniform as randUniform
from math import pi
import matplotlib.pyplot as plt
"""
TASK 2 - QOSF SCREENING TEST
Implement a circuit that returns |01> and |10> with equal probability.
Requirements :
1) The circuit should consist only of CNOTs, RXs and RYs. 
2) Start from all parameters in parametric gates being equal to 0 or randomly chosen. 
3) You should find the right set of parameters using gradient descent (you can use more advanced optimization methods if you like). 
4) Simulations must be done with sampling (i.e. a limited number of measurements per iteration) and noise. 

A) Compare the results for different numbers of measurements: 1, 10, 100, 1000. 

Bonus question:
How to make sure you produce state |01>+|10> and not |01>-|10>?

(Actually for more careful readers, the “correct” version of this question is posted below:
How to make sure you produce state  |01⟩+|10⟩  and not any other combination of |01>+e(i*phi)|10⟩ (for example |01⟩-|10⟩)?)
"""

"""
From experience already, it is easy to see that the state |01>+|10> and |01>-|10> are just like the typical Bell pairs 
that, when measured, result in the state |01> 50% of the time and |10> the other 50% of the time, and that to achieve this, 
all that is needed is a Hadamard gate, an X gate and a CNOT.

However, we can think of this from a programmer's perspective. Thinking of CNOT as an "IF" gate and H as a rand() 
function of 50/50 chance between 0 and 1, you'd want to make a system that says "IF first qubit is 0, THEN second qubit 
is 1" and "IF first qubit is 1, THEN second qubit is 0" and then set the first qubit to randomly be either a 0 or 1. 
This will logically result in a 50/50 chance of getting |01> and |10>. There is a more mathematical explanation that reaches
the result more formally, but it's quite difficult to show in just a text format.

However, the task requires us to use only RX and/or RY and CNOT (1), without a Hadamard gate. Again, from experience, one 
already knows that the H gate can be decomposed into RX(pi) and RY(pi/2), for example. Fortunately, since we have access
to the RY gate, we can directly map the qubit from |0> to |+> (aka |0>+|1>) or |-> (aka |0>-|1>), so we don't need all 
of the properties of the Hadamard gate (such as the involutory property) to achieve our desired goal, just the fact that 
it makes our first qubit be |0> or |1> 50% of the time each at measurement (since it is the control qubit, the state of
the qubit is not affected by the CNOT). So the first qubit will only have an RY gate before it reaches the CNOT and then 
finally measurement.

Now, with just an RY gate shifting the phase of the first qubit by either pi/2 or 3*pi/2 and running both qubits through 
a CNOT, the measurement results will turn out |00> and |11>. This is because the second qubit (states are read from right 
to left) is set to |0>. To fix this, all we need is to flip it to |1> before reaching the CNOT to achieve the desired 
states |10> and |01>. We'll use the RX gate, although RY would have also worked perfectly for our situation.

     ┌───────────────┐     ┌─┐   
q_0: ┤ RY(params[0]) ├──■──┤M├───
     ├───────────────┤┌─┴─┐└╥┘┌─┐
q_1: ┤ RX(params[1]) ├┤ X ├─╫─┤M├
     └───────────────┘└───┘ ║ └╥┘
c: 2/═══════════════════════╩══╩═

It would have been easier to apply the optimization method on a circuit with no noise in order to be much more precise, 
but once more, the exercise requires a little more work, wanting us to guide the algorithm with noise and a limited 
number of measurements (4).

The values (params[i]) will be initially randomly set (2), and then from there we will apply gradient descent (3) to modify 
these values and turn them into very good approximations of the correct ones (pi/2 to get |+> or 3*pi/2 to get |-> through 
the RY gate of the first qubit, and pi to get |1> through the RX gate of the second qubit).
"""

class GradientDescent:
	def __get_counts(self, params: List[float or int], shots: int = 1000) -> dict:
		"""
		Here we run the circuit according to the given parameters for each gate and return the counts for each state.

		:param params: List of the parameters of the RY and RX gates of the circuit.
		:param shots: Total number of shots the circuit must execute
		"""
		# Error probabilities
		prob_1 = 0.001  # 1-qubit gate
		prob_2 = 0.01  # 2-qubit gate

		# Depolarizing quantum errors
		error_1 = noise.depolarizing_error(prob_1, 1)
		error_2 = noise.depolarizing_error(prob_2, 2)

		# Add errors to noise model
		noise_model = noise.NoiseModel()
		noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2'])
		noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

		# Get basis gates from noise model
		basis_gates = noise_model.basis_gates

		# Make a circuit
		circ = QuantumCircuit(2, 2)

		# Set gates and measurement
		circ.ry(params[0], 0)
		circ.rx(params[1], 1)
		circ.cx(0, 1)
		circ.measure([0, 1], [0, 1])

		# Perform a noisy simulation and get the counts
		# noinspection PyTypeChecker
		result = execute(circ, Aer.get_backend('qasm_simulator'),
						 basis_gates=basis_gates,
						 noise_model=noise_model, shots=shots).result()
		counts = result.get_counts(0)

		return counts

	def __get_cost_vector(self, counts: dict) -> List[float]:
		"""
		This function simply gives values that represent how far away from our desired goal we are. Said desired goal is that
		we get as close to 0 counts for the states |00> and |11>, and as close to 50% of the total counts for |01> and |10>
		each.

		:param counts: Dictionary containing the count of each state
		:return: List of ints that determine how far the count of each state is from the desired count for that state:
					-First element corresponds to |00>
					-Second element corresponds to |01>
					-Third element corresponds to |10>
					-Fourth element corresponds to |11>
		"""
		# First we get the counts of each state. Try-except blocks are to avoid errors when the count is 0.
		try:
			a = counts['00']
		except KeyError:
			a = 0
		try:
			b = counts['01']
		except KeyError:
			b = 0
		try:
			c = counts['10']
		except KeyError:
			c = 0
		try:
			d = counts['11']
		except KeyError:
			d = 0

		# We then want the total number of shots to know what proportions we should expect
		totalShots = a + b + c + d

		# We return the absolute value of the difference of each state's observed and desired counts
		# Other systems to determine how far each state count is from the goal exist, but this one is simple and works well
		return [abs(a - 0), abs(b - totalShots / 2), abs(c - totalShots / 2), abs(d - 0)]

	def __done_descending(self, costVector: List[float or int], shots: int, costTolerance: float) -> bool:
		"""
		Determines whether we should stop the algorithm or not
		:param costVector: List of ints that determine how far the count of each state is from the desired count for that state
		:param shots: Number of shots taken by the circuit
		:param costTolerance: Float value that determines the error margin that will be allowed.
		:return: True if all costs are lower than costTolerance * shots, False otherwise
		"""
		if not 1 >= costTolerance > 0:
			raise ValueError("costTolerance must be in the range (0, 1]")

		for cost in costVector:
			if cost > costTolerance * shots:
				return False
		return True

	def __get_gradient(self, counts: dict) -> List[int]:
		"""
		The gradient will determine how strongly we will modify the values in params[] and in which direction.

		Let's remember again what we want each qubit to do:
			1) We want the first qubit to be measured as |0> 50% of the time, with the other 50% of the time being |1>, obviously.
			2) We want the second qubit to always be |1> before it reaches the CNOT.
		Remember that the counts give us information AFTER the CNOT.

		We can easily determine how far the first parameter (the one controlling the phase of the first qubit) is from its
		desired goal by seeing how often it is a |0> or a |1> once it has been measured. This can be inferred by adding the
		counts of |00> and |10>, we get the count of the first qubit being |0>. Of course, adding the counts of |01> and |11>
		gives us how often the first qubit is |1>.

		With that information, we can determine in which direction and how strongly we should modify the values in params[i]
		with a function such as (count |00> + count |10>) - (count |01> + count |11>). The more often |0> appears than |1>,
		the greater the output will be on the positive side, and the more often |1> appears than |0>, the greater the output
		will be on the negative side. This is what we will use as the gradient function for the parameter of the first qubit.

		The second qubit is a little trickier to think about, but the function is just as simple. We said we wanted the second
		qubit to always be |1> BEFORE it reaches the CNOT, but we don't have access to measurement before the CNOT. Then, how
		do we tell how we should modify the parameter for the RX gate on the second qubit?

		Going back to the other big comment in this file, we know that if there were no RX (or RY) gate on the second qubit,
		that the output will always be |00> or |11>, since the CNOT only flips the second qubit if the first one is flipped.
		This way, we know that if we are getting counts of |00> and |11> that are way too high above 0, the parameter of the
		gate is not where it should be. Therefore, the higher the total count of |00> and |11>, the stronger the change in
		phase we need to make, and a function that represents that well is simply the total count of |00> and |11> itself.

		There is a small problem with this function, and that is that it's only one-directional (we will never get a negative
		value for this function). This means that perhaps it would be optimal to decrease the value of the parameter at a given
		moment, rather than to increase it, but the function will always guide us to increase it. This isn't really a huge issue,
		since the phase is modulo 2*pi. What matters is that it tells us to increase the value at the right pace at every given
		moment: when we are really far from the goal, take big steps, and as we get closer to the goal, take smaller and smaller
		steps. The function proposed earlier does that very well.

		One extra thing to note: the function of the gradient of the second qubit highly depends on the effectiveness of the
		function of the first qubit. If the first one isn't working, the second one is flying blind.

		With these two functions dictating how strongly we should alter each parameter and tweaking some constants such as
		the learning rate that we will use later on, we will obtain the correct values for each parameter fairly quickly.

		:param counts: Dictionary containing the count of each state
		:return: List of ints that are a representation of how intensely we must change the values of params[]
					-params[0] corresponds to the phase of the RY gate of the first qubit
					-params[1] corresponds to the phase of the RX gate of the second qubit
		"""
		try:
			a = counts['00']
		except KeyError:
			a = 0
		try:
			b = counts['01']
		except KeyError:
			b = 0
		try:
			c = counts['10']
		except KeyError:
			c = 0
		try:
			d = counts['11']
		except KeyError:
			d = 0

		# We then want the total number of shots to know what proportions we should expect
		totalShots = a + b + c + d

		"""
		Let's quickly think about a way of improving the gradient than just the difference or addition of counts.
		
		First, let's note that without any modifications, the range of the gradients are [-0.5, 0.5] and [0, 1] for params[0]
		and params[1], respectively. If our values turned out to be at the extreme side of the undesired, then we know very 
		well by how much we should change the phase.
		
		For example, if for params[1] we got a gradient of 1, that means that params[1] is somewhere around |0> and so it needs
		a shift of pi to be |1>. To get to that point very quickly, we can just multiply the value of the gradient by pi. 
		That way, whenever the second parameter is on the complete opposite end of where it should be, we can move it very 
		quickly to where it should. When I only modify the second gradient by multiplying it by pi, this often reduces the 
		number of steps taken by the algorithm significantly (for example, from an average of 20 steps to an average of 6 steps).
		
		Now, you would think that the same goes for the first parameter: if the gradient is ±0.5, then params[0] is 
		probably around 0 or pi, and we want to move it to pi/2 or 3*pi/2, so we need the gradient to be ±pi/2. We take 
		advantage of the 0.5 and multiply it by pi in order to make the gradient equal to ±pi/2 whenever it is at 
		either extreme. But this is not the case. For some reason that I don't fully comprehend, it makes the algorithm take
		more steps in general, especially when combined with the modification of the second gradient value. 
		
		I suspect that it has to do with the fact that the second gradient value has a greater "tolerable margin of error" 
		than the first gradient value. That is, params[1] could be anywhere from 3*pi/4 to 5*pi/4 and still output often 
		enough it's intended goal, whilst that range of 2*pi/4 (or pi/2) means the opposite output of what is desired. Either
		way, I'd love to discuss it further with my mentor.
		"""
		return [((b + d) - (a + c))/totalShots, pi * (a + d)/totalShots]

	def __update_params(self, params: List[float], gradient: List[int], learningRate: float or int) -> None:
		"""
		Here we update the parameters according to the gradient. A very simple update function is just subtracting the gradient,
		and modulating the intensity of said gradient with a learning rate value.

		:param params: List of the parameters of the RY and RX gates of the circuit.
		:param gradient: List of values which represent how intensely we should modify each parameter
		:param learningRate: Float or int value to modulate the gradient
		"""
		for i in range(len(params)):  # For every parameter value
			params[i] -= learningRate * gradient[i]  # Subtract a modulated version of the value of the gradient

			# Make value mod 2*pi for the sake of not using unnecessarily big numbers. It's not necessary, but it's cleaner.
			while params[i] < 0:
				params[i] += 2 * pi
			params[i] = params[i] % (2 * pi)


	def start(self, shots: int =1000, costTolerance: float = 0.01, learningRate: float or int = 1) -> None:
		"""
		Main method that does the entire gradient descent algorithm.

		:param costTolerance: Float value that determines the error margin that will be allowed.
		:param learningRate: Float or int value to modulate the gradient
		:param shots: Total number of shots the circuit must execute
		"""
		# First, we start with some random parameters
		params = [randUniform(0, 2 * pi), randUniform(0, 2 * pi)]

		# Initialize variables
		descending = True  # Symbolic, makes it easier for reader to follow
		counts = None
		costVector = None

		print("START PARAMS: " + str(params))  # Print to console the parameters we will begin with

		steps = 0  # Keep track of how many steps were taken to descend
		while descending:
			# Get the initial counts that result from these parameters
			counts = self.__get_counts(params, shots)

			# Find the cost of these parameters given the results they have produced
			costVector = self.__get_cost_vector(counts)

			# Determine whether the cost is low enough to stop the algorithm
			if self.__done_descending(costVector, shots, costTolerance):
				descending = False  # Symbolic, makes it easier for reader to follow
				break

			# Calculate the gradient
			gradient = self.__get_gradient(counts)

			# Update the params according to the gradient
			self.__update_params(params, gradient, learningRate)
			steps += 1  # Recording the number of steps taken

			# Show current situation
			print("\tCOUNTS: " + str(counts)
				  + "\n\tCOST VECTOR: " + str(costVector)
				  + "\n\tGRADIENT: " + str(gradient)
				  + "\n\tUPDATED PARAMS: " + str(params) + "\n")

		# Print the obtained results
		print("FINAL RESULTS:"
		+ "\n\tCOUNTS: " + str(counts)
		+ "\n\tCOST VECTOR" + str(costVector)
		+ "\n\tPARAMS: " + str(params)
		+ "\nSteps taken: " + str(steps))

# Execution of the main method
gd = GradientDescent()
gd.start()

"""
A) COMPARE THE RESULTS FOR DIFFERENT NUMBERS OF MEASUREMENTS: 1, 10, 100, 1000.

First of all, just one shot will not produce enough information to properly guide the gradient descent. It simply
doesn't tell us at all what is the probability of each state being measured, and so the algorithm will go on forever,
trying to make these two states appear an equal number of times when we are only allowing one state to appear at all. Two
shots would at least give the algorithm a chance to stop, but the results obtained will, very likely, be incorrect.

Ten shots, however, is a lot better, since we have a bigger sample from which we can infer the probabilities of each
state. Despite this, the data obtained isn't really statistically significant, and so the parameters that the algorithm
lands on will have a decently large margin of error. To solve that, we need a bigger sample.

With one hundred shots, we finally have a fairly decent size of a sample to more accurately determine what are the 
probabilities of each state, and so the margin of error will decrease as the algorithm adjusts the parameters to the 
correct ones.

One thousand shots is even better for the same reason: a bigger sample. When put into practice, that is exactly the case: 
the parameters obtained are much more accurate. We can obviously notice a pattern here, so why not go on increasing the
number of shots?

Well, as we increase the number of shots, we see that the noise becomes an unavoidable issue. It doesn't matter that we 
are using exactly the correct parameters, noise exacerbates the counts of the states that we don't expect from our circuit. 
This will throw off our algorithm, trying to readjust itself again and again when the value it had to begin with was already 
perfect, not knowing that the fault is of the imperfect world outside.

------------------------------------------------------------------------------------------------------------------------

BONUS QUESTION:
How to make sure you produce state |01>+|10> and not |01>-|10>?
(Actually for more careful readers, the “correct” version of this question is posted below:
How to make sure you produce state  |01⟩+|10⟩  and not any other combination of |01>+e(i*phi)|10⟩ (for example |01⟩-|10⟩)?)


To answer this question, let's first understand what a state is.

The state of a combination of qubits is a collection of conditions that the arrangement of qubits meets. These conditions
are in relation to a certain basis, and more specifically, a linear combination of said basis states. For example, the state
|0> is a linear combination of the basis states α|0>+β|1>, where α=1 and β=0 (these states are written in ket notation, 
but they can also be written in vector notation too). Said coefficients must comply with the rule that the sum of the 
squares of the coefficients must be equal to 1, since the square of the absolute value of each coefficient determines the
probability of measuring that basis state (Born's rule).

These coefficients don't need to be positive or even real numbers. They can also be completely imaginary or complex, and
so the state |0>-i|1> is perfectly valid, as well as |0>+|1>, |0>-|1> and |0>+i|1>. However, at measurement, they will 
all return |0> or |1> 50% of the time each. So how can we differentiate between these states? We won't be able to tell 
with measurement alone, since they all have the same probabilities. The only way to differentiate them is to put them 
through gates that modify the coefficients of the states. 

So let's go back to the original question: how to ensure that we are producing |01>+|10> in our circuit and not any other 
state that has the same probability output?

As established before, we need more gates to differentiate between the two. Without getting into excessive detail, a CNOT
followed by a CNOT is the same as nothing, so that way we can measure each qubit independently. Then we can reconstruct a
Hadamard gate with RX and RY, mapping the state |+> to |0> and |-> to |1>. What we would find is that the first qubit is 
in either state |+> or |->, depending on what we measured, due to the fact that we put an RY gate there at the start. 

Again, skipping over some maths, we would see that the |+> state is the one that allows the state |01>+|10> to exist once 
combined with the CNOT, and the |-> state allows the |01>-|10> state to exist when combined with the CNOT. So we know that 
we must limit the first qubit to only being in the |+> state. We don't have to worry about any other states producing the 
same output probabilities since we have limited the first qubit to just rotation around the Y axis, and so the only "threat" 
was the |-> state.

Now, how do we limit the first qubit to just being |+> and not |->? If the gradient descent had as input the parameters
themselves, this would be solved a lot more quickly, but I doubt that is the point of the exercise. So, since we only have
the counts of the states, what can we do?

We know that, with a qubit initialized at |0>, a Y rotation of pi/2 will map it to |+> and a Y rotation of 3*pi/2 will 
map it to |->. For a second, let's think of the case where we see that the first qubit has more 0s than 1s appearing. That 
means that the phase is around either pi/3 or 5*pi/3 approximately. If we increase the phase of that gate (params[0]), in 
the case where it was pi/3, we'll see that shift to around pi/2, and in the case of 5*pi/3, we'll see that shift to around 
2*pi==0. So in the first case, the output of 0s and 1s will be the same, while in the second case the output of 0s will 
have increased even more.

We can see then that whenever there are more 0s than 1s when measuring the first qubit, we must tell our algorithm to 
increase the value of the parameter, and whenever there are more 1s than 0s, we tell it to decrease it. This way, the 
value will almost always end up being pi/2. This is exactly what the code above does. The only situation where the 
algorithm will give us a params[0] value of 3*pi/2 is if params[0] is already 3*pi/2 and params[1] is pi already from 
the start, because then it will already give us the correct counts and the algorithm will not be inclined to modify any 
parameter, returning the undesired values.
"""