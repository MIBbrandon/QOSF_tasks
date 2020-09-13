# QOSF Screening Tasks  
## Task 2  
  
### **TASK 2 - QOSF SCREENING TEST**
**Implement a circuit that returns |01> and |10> with equal probability.**  

***Requirements :***
 1. **The circuit should consist only of CNOTs, RXs and RYs.**
 2. **Start from all parameters in parametric gates being equal to 0 or
    randomly chosen.**
 3. **You should find the right set of parameters using gradient descent (you can use more advanced optimization methods if you like).**
 4. **Simulations must be done with sampling (i.e. a limited number of measurements per iteration) and noise.**

**Compare the results for different numbers of measurements: 1, 10, 100, 1000.**

**Bonus question:  
How to make sure you produce state |01>+|10> and not |01>-|10>?**  
  
**(Actually for more careful readers, the “correct” version of this question is posted below:**  
**How to make sure you produce state  |01⟩+|10⟩ and not any other combination of |01>+e(i\*phi)|10⟩ (for example |01⟩-|10⟩)?)**

From experience already, it is easy to see that the state |01>+|10> and |01>-|10> are just like the typical Bell pairs that, when measured, result in the state |01> 50% of the time and |10> the other 50% of the time, and that to achieve this, all that is needed is a Hadamard gate, an X gate and a CNOT.  
  
However, we can think of this from a programmer's perspective. Thinking of CNOT as an "IF" gate and H as a rand() function of 50/50 chance between 0 and 1, you'd want to make a system that says "IF first qubit is 0, THEN second qubit is 1" and "IF first qubit is 1, THEN second qubit is 0" and then set the first qubit to randomly be either a 0 or 1. This will logically result in a 50/50 chance of getting |01> and |10>. There is a more mathematical explanation that reaches the result more formally, but it's quite difficult to show in just a text format.  
  
However, the task requires us to use only RX and/or RY and CNOT (1), without a Hadamard gate. Again, from experience, one already knows that the H gate can be decomposed into RX(pi) and RY(pi/2), for example. Fortunately, since we have access to the RY gate, we can directly map the qubit from |0> to |+> (aka |0>+|1>) or |-> (aka |0>-|1>), so we don't need all of the properties of the Hadamard gate (such as the involutory property) to achieve our desired goal, just the fact that it makes our first qubit be |0> or |1> 50% of the time each at measurement (since it is the control qubit, the state of the qubit is not affected by the CNOT). So the first qubit will only have an RY gate before it reaches the CNOT and then finally measurement.  
  
Now, with just an RY gate shifting the phase of the first qubit by either pi/2 or 3*pi/2 and running both qubits through a CNOT, the measurement results will turn out |00> and |11>. This is because the second qubit (states are read from right to left) is set to |0>. To fix this, all we need is to flip it to |1> before reaching the CNOT to achieve the desired states |10> and |01>. We'll use the RX gate, although RY would have also worked perfectly for our situation.  

![Circuit image](https://github.com/MIBbrandon/QOSF_tasks/blob/master/circuit_image.png)

It would have been easier to apply the optimization method on a circuit with no noise in order to be much more precise, but once more, the exercise requires a little more work, wanting us to guide the algorithm with noise and a limited number of measurements (4).  
  
The values (params\[i]) will be initially randomly set (2), and then from there we will apply gradient descent (3) to modify these values and turn them into very good approximations of the correct ones (pi/2 to get |+> or 3*pi/2 to get |-> through the RY gate of the first qubit, and pi to get |1> through the RX gate of the second qubit).


### COMPARE THE RESULTS FOR DIFFERENT NUMBERS OF MEASUREMENTS: 1, 10, 100, 1000
First of all, just one shot will not produce enough information to properly guide the gradient descent. It simply doesn't tell us at all what is the probability of each state being measured, and so the algorithm will go on forever, trying to make these two states appear an equal number of times when we are only allowing one state to appear at all. Two shots would at least give the algorithm a chance to stop, but the results obtained will, very likely, be incorrect.  
  
Ten shots, however, is a lot better, since we have a bigger sample from which we can infer the probabilities of each state. Despite this, the data obtained isn't really statistically significant, and so the parameters that the algorithm lands on will have a decently large margin of error. To solve that, we need a bigger sample.  
  
With one hundred shots, we finally have a fairly decent size of a sample to more accurately determine what are the probabilities of each state, and so the margin of error will decrease as the algorithm adjusts the parameters to the correct ones.  
  
One thousand shots is even better for the same reason: a bigger sample. When put into practice, that is exactly the case: the parameters obtained are much more accurate. We can obviously notice a pattern here, so why not go on increasing the number of shots?  
  
Well, as we increase the number of shots, we see that the noise becomes an unavoidable issue. It doesn't matter that we are using exactly the correct parameters, noise exacerbates the counts of the states that we don't expect from our circuit. This will throw off our algorithm, trying to readjust itself again and again when the value it had to begin with was already perfect, not knowing that the fault is of the imperfect world outside.

### BONUS QUESTION 
**How to make sure you produce state |01>+|10> and not |01>-|10>?  
(Actually for more careful readers, the “correct” version of this question is posted below:  
How to make sure you produce state  |01⟩+|10⟩ and not any other combination of |01>+e(i\*phi)|10⟩ (for example |01⟩-|10⟩)?)**

To answer this question, let's first understand what a state is.  
  
The state of a combination of qubits is a collection of conditions that the arrangement of qubits meets. These conditions are in relation to a certain basis, and more specifically, a linear combination of said basis states. For example, the state |0> is a linear combination of the basis states α|0>+β|1>, where α=1 and β=0 (these states are written in ket notation, but they can also be written in vector notation too). Said coefficients must comply with the rule that the sum of the squares of the coefficients must be equal to 1, since the square of the absolute value of each coefficient determines the probability of measuring that basis state (Born's rule).  
  
These coefficients don't need to be positive or even real numbers. They can also be completely imaginary or complex, and so the state |0>-i|1> is perfectly valid, as well as |0>+|1>, |0>-|1> and |0>+i|1>. However, at measurement, they will all return |0> or |1> 50% of the time each. So how can we differentiate between these states? We won't be able to tell with measurement alone, since they all have the same probabilities. The only way to differentiate them is to put them through gates that modify the coefficients of the states. So let's go back to the original question: how to ensure that we are producing |01>+|10> in our circuit and not any other state that has the same probability output?  
  
As established before, we need more gates to differentiate between the two. Without getting into excessive detail, a CNOT followed by a CNOT is the same as nothing, so that way we can measure each qubit independently. Then we can reconstruct a Hadamard gate with RX and RY, mapping the state |+> to |0> and |-> to |1>. What we would find is that the first qubit is in either state |+> or |->, depending on what we measured, due to the fact that we put an RY gate there at the start. Again, skipping over some maths, we would see that the |+> state is the one that allows the state |01>+|10> to exist once combined with the CNOT, and the |-> state allows the |01>-|10> state to exist when combined with the CNOT. So we know that we must limit the first qubit to only being in the |+> state. We don't have to worry about any other states producing the same output probabilities since we have limited the first qubit to just rotation around the Y axis, and so the only "threat" was the |-> state.  
  
Now, how do we limit the first qubit to just being |+> and not |->? If the gradient descent had as input the parameters themselves, this would be solved a lot more quickly, but I doubt that is the point of the exercise. So, since we only have the counts of the states, what can we do?  
  
We know that, with a qubit initialized at |0>, a Y rotation of pi/2 will map it to |+> and a Y rotation of 3\*pi/2 will map it to |->. For a second, let's think of the case where we see that the first qubit has more 0s than 1s appearing. That means that the phase is around either pi/3 or 5*pi/3 approximately. If we increase the phase of that gate (params\[0]), in the case where it was pi/3, we'll see that shift to around pi/2, and in the case of 5\*pi/3, we'll see that shift to around 2\*pi==0. So in the first case, the output of 0s and 1s will be the same, while in the second case the output of 0s will have increased even more.  
  
We can see then that whenever there are more 0s than 1s when measuring the first qubit, we must tell our algorithm to increase the value of the parameter, and whenever there are more 1s than 0s, we tell it to decrease it. This way, the value will almost always end up being pi/2. This is exactly what the code in task_2.py does. The only situation where the algorithm will give us a params\[0] value of 3\*pi/2 is if params\[0] is already 3*pi/2 and params\[1] is pi already from the start, because then it will already give us the correct counts and the algorithm will not be inclined to modify any parameter, returning the undesired values.