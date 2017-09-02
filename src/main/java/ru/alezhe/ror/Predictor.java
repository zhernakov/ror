/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ru.alezhe.ror;

/**
 *
 * @author Aleksandr
 */
public interface Predictor {
    
    public double getNextStepVariance();
    
    public double calcVaR(double p);
}
