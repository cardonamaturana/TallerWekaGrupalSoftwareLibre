/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package tallerwekagrupalsoftwarelibre;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author USUARIO
 */
public class TallerWekaGrupalSoftwareLibre {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {

        try {
            String ruta="regresion.arff";
            //Cargar las instancias
            Instances instancias= new Instances(new BufferedReader(new FileReader(ruta)));
            instancias.setClassIndex(1);
            
            //Crear el modelo de regresión
            LinearRegression lr=new LinearRegression();
            lr.buildClassifier(instancias);
            System.out.println("LR " + lr);
            System.out.println("Coef "+Arrays.toString(lr.coefficients()));
            
            //Evaluación del modelo
            Evaluation ev=new Evaluation(instancias);
            ev.crossValidateModel(lr, instancias, 10, new Random(1), new String[]{});
            System.out.println(""+ev.toSummaryString());
            
            Scanner teclado=new Scanner(System.in);
            System.out.println("Ingrese el valor de x ");
            int x=teclado.nextInt(); 
            double coef[]=lr.coefficients(); 
            double y=x*coef[0]+coef[2];
            System.out.println("El valor de y es "+y);     
            
            
            Instance i=new Instance(2);
            i.setDataset(instancias);
            i.setValue(0, 300);
            System.out.println("Predicción"+lr.classifyInstance(i));
            
            
            
                        
        } catch (FileNotFoundException ex) {
            Logger.getLogger(TallerWekaGrupalSoftwareLibre.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(TallerWekaGrupalSoftwareLibre.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(TallerWekaGrupalSoftwareLibre.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
