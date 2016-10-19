#!/usr/bin/env swift
//
//  main.swift
//  lecture1
//
//  Created by Lane Schwartz on 2016-09-12.
//  Copyright © 2016 Lane Schwartz. All rights reserved.
//

import Foundation

//print("Hello, World!")

func applyModel(weights:[Float], withInput x:Float) -> Float {
  return weights[0] + weights[1]*x
}

func summation (accumulator:Float, newValue:Float) -> Float {
  return accumulator + newValue
}

func J(θ:[Float], x:[Float], y:[Float]) -> Float {
  let m = y.count
  
  return 1.0/Float(2*m) *
    (0..<m).map{ i in // for each i in 1 to m
      applyModel(weights: θ, withInput: x[i]) - y[i] // calculate h_θ(x_i) - y_i for all i
      }.map{ value in
        pow(value, 2.0) // now square each value
      }.reduce(0.0, summation) // finally, sum all values
  
}

func gradientDescent(theta θ:[Float], x:[Float], y:[Float], alpha α:Float = 0.1, epsilon ε:Float = 0.001) -> [(Float, [Float])] {
  
  func updateTheta(_ θ:[Float]) -> [Float] {
    
    let m = y.count
    
    let newθ_0 : Float = θ[0] - α/Float(m) *
      (0..<m).map{ i in // for each i in 1 to m
        applyModel(weights: θ, withInput: x[i]) - y[i] // calculate h_θ(x_i) - y_i for all i
        }.reduce(0.0, summation) // finally, sum all values
    
    let part1 = α/Float(m)
    let part2 = (0..<m).map{ i in // for each i in 1 to m
      (applyModel(weights: θ, withInput: x[i]) - y[i]) * x[i] // calculate (h_θ(x_i) - y_i) * x_i for all i
    }
    let part3 = part2.reduce(0.0, summation)
    
    let newθ_1 : Float = θ[1] - part1 * part3// finally, sum all values
    
    //print("theta[1] part2: \(part3)")
    
    return [newθ_0, newθ_1]
  }
  
  func descend(θ:[Float], previousError:Float, errors:[(Float, [Float])]) -> [(Float, [Float])] {
    
    let newθ = updateTheta(θ)
    let currentError = J(θ: newθ, x:x, y:y)
    if (currentError.isInfinite || abs(previousError - currentError) <= ε) {
      //print("exiting early \(previousError) \(currentError)")
      return errors
    } else {
      //print("recursing \(previousError - currentError) \(previousError) \(currentError) \(newθ)")
      return descend(θ: newθ, previousError:currentError, errors: errors + [(currentError, newθ)])
    }
    
  }
  
  let initialError = J(θ:θ, x:x, y:y)
  
  return descend(θ: θ, previousError: initialError, errors: [(initialError, θ)])
  
}

let theta:[Float] = [0.0, 0.0]
//let theta:[Float] = [-3.8958, 1.1930]

let x:[Float] = [6.1101, 5.5277, 8.5186, 7.0032, 5.8598, 8.3829, 7.4764, 8.5781, 6.4862, 5.0546, 5.7107, 14.164, 5.734, 8.4084, 5.6407, 5.3794, 6.3654, 5.1301, 6.4296, 7.0708, 6.1891, 20.27, 5.4901, 6.3261, 5.5649, 18.945, 12.828, 10.957, 13.176, 22.203, 5.2524, 6.5894, 9.2482, 5.8918, 8.2111, 7.9334, 8.0959, 5.6063, 12.836, 6.3534, 5.4069, 6.8825, 11.708, 5.7737, 7.8247, 7.0931, 5.0702, 5.8014, 11.7, 5.5416, 7.5402, 5.3077, 7.4239, 7.6031, 6.3328, 6.3589, 6.2742, 5.6397, 9.3102, 9.4536, 8.8254, 5.1793, 21.279, 14.908, 18.959, 7.2182, 8.2951, 10.236, 5.4994, 20.341, 10.136, 7.3345, 6.0062, 7.2259, 5.0269, 6.5479, 7.5386, 5.0365, 10.274, 5.1077, 5.7292, 5.1884, 6.3557, 9.7687, 6.5159, 8.5172, 9.1802, 6.002, 5.5204, 5.0594, 5.7077, 7.6366, 5.8707, 5.3054, 8.2934, 13.394, 5.4369]

let y:[Float] = [17.592, 9.1302, 13.662, 11.854, 6.8233, 11.886, 4.3483, 12, 6.5987, 3.8166, 3.2522, 15.505, 3.1551, 7.2258, 0.71618, 3.5129, 5.3048, 0.56077, 3.6518, 5.3893, 3.1386, 21.767, 4.263, 5.1875, 3.0825, 22.638, 13.501, 7.0467, 14.692, 24.147, -1.22, 5.9966, 12.134, 1.8495, 6.5426, 4.5623, 4.1164, 3.3928, 10.117, 5.4974, 0.55657, 3.9115, 5.3854, 2.4406, 6.7318, 1.0463, 5.1337, 1.844, 8.0043, 1.0179, 6.7504, 1.8396, 4.2885, 4.9981, 1.4233, -1.4211, 2.4756, 4.6042, 3.9624, 5.4141, 5.1694, -0.74279, 17.929, 12.054, 17.054, 4.8852, 5.7442, 7.7754, 1.0173, 20.992, 6.6799, 4.0259, 1.2784, 3.3411, -2.6807, 0.29678, 3.8845, 5.7014, 6.7526, 2.0576, 0.47953, 0.20421, 0.67861, 7.5435, 5.3436, 4.2415, 6.7981, 0.92695, 0.152, 2.8214, 1.8451, 4.2959, 7.2029, 1.9869, 0.14454, 9.0551, 0.61705]


let results = gradientDescent(theta: theta, x: x, y: y, alpha: 0.01, epsilon: 0.0)
print(results.last!)
