using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Collections;

namespace BP_TrainningAFunction
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        int TrainingDataAmount = 21;
        int TestingDataAmount;
        Stack pub = new Stack();
        Stack pub_w = new Stack();
        int neuron_amount;
        int neuron2_amount;
        decimal LearningRate = (decimal)10;

        double x;
        Random crandom = new Random();

        private void button1_Click(object sender, EventArgs e)
        {
            decimal[,] training_set = new decimal[2, TrainingDataAmount];
            for (decimal i = 0; i <= 4; i += (decimal)0.2) {
                training_set[0, (int)(i * 5)] = i;
                training_set[1, (int)(i * 5)] = (decimal)Math.Exp(-(double)i) * (decimal)Math.Sin((double)i*(double)3) + 0.3M;

            }
            pub.Push(training_set);
        }


        private void button2_Click(object sender, EventArgs e)
        {
            decimal[,] training_set = (decimal[,])pub.Pop();
            neuron_amount = Convert.ToInt32(textBox1.Text);
            neuron2_amount = Convert.ToInt32(textBox3.Text);


            decimal[] y1 = new decimal[neuron_amount];
            decimal[] y1_activity = new decimal[neuron_amount];
            decimal[] y2 = new decimal[neuron2_amount];
            decimal[] y2_activity = new decimal[neuron2_amount];
            decimal y3, y3_activity;

            decimal[] w1 = new decimal[neuron_amount];
            decimal[,] w2 = new decimal[neuron_amount, neuron2_amount];
            decimal[] w3 = new decimal[neuron2_amount];

            decimal[] w1_change = new decimal[neuron_amount];
            decimal[,] w2_change = new decimal[neuron_amount, neuron2_amount];
            decimal[] w3_change = new decimal[neuron2_amount];

            decimal delta;
            decimal[] delta_hidden2 = new decimal[neuron2_amount];
            decimal[] delta_hidden1 = new decimal[neuron_amount];
            decimal error;
            decimal SumSquareError = 100;
            int RealEpochs = 0;

            decimal DesireError = Convert.ToDecimal(textBox4.Text);

            for (int k = 0; k < neuron_amount; k++)
            {
                x = crandom.NextDouble()*4-2; //給予亂數
                w1[k] = (decimal)x;
            }

            for (int k = 0; k < neuron_amount; k++)
            {
                for (int j = 0; j < neuron2_amount; j++) {
                    x = crandom.NextDouble()*4-2; //給予亂數
                    w2[k, j] = (decimal)x;                    
                }
            }

            for (int k = 0; k < neuron2_amount; k++)
            {
                x = crandom.NextDouble()*4-2; //給予亂數
                w3[k] = (decimal)x;                
            }

            //權重初值：0.5
            //for (int i=0; i < neuron_amount; i++)
            //{
            //    w1[i] = (decimal)0.1*i;
            //    w2[i] = (decimal)0.5*i;
            //}
            int times = 100000;
            //for (int s = 0; s < times; s++) { 
            while (SumSquareError > DesireError) {
                SumSquareError = 0;
                for (int i = 0; i < TrainingDataAmount; i++)
                {
                    for (int j = 0; j < neuron_amount; j++) {
                        y1[j] = training_set[0, i] * w1[j];
                        y1_activity[j] = 1/(1+(decimal)Math.Exp(-(double)y1[j]));
                    }                       
                                        
                    for (int j = 0; j < neuron2_amount; j++)
                    {
                        y2[j] = 0;               
                        for (int k = 0; k < neuron_amount; k++)
                            y2[j] += y1_activity[k] * w2[k,j];
                        y2_activity[j] = 1 / (1 + (decimal)Math.Exp(-(double)y2[j])); //激勵y2
                    }

                    y3 = 0;
                    for (int j = 0; j < neuron2_amount; j++)
                        y3 += y2_activity[j] * w3[j];
                    y3_activity = 1 / (1 + (decimal)Math.Exp(-(double)y3)); //激勵y3
                    
                    error = training_set[1, i] - y3_activity;
                    SumSquareError += error * error;
                    delta = error * y3_activity * (1 - y3_activity);

                    for (int j = 0; j < neuron2_amount; j++)
                    {
                        delta_hidden2[j] = delta * w3[j] * y2_activity[j] * (1 - y2_activity[j]);
                    }

                    for (int j = 0; j < neuron_amount; j++)
                    {
                        delta_hidden1[j] = 0;
                        for (int k = 0; k < neuron2_amount; k++)
                            delta_hidden1[j] += delta_hidden2[k] * w2[j,k];
                        delta_hidden1[j] *= y1_activity[j] * (1 - y1_activity[j]);
                    }

                    for (int j = 0; j < neuron2_amount; j++)
                    {
                        w3_change[j] += y2_activity[j] * LearningRate * delta;
                    }

                    for (int j = 0; j < neuron_amount; j++)
                    {
                        for (int k = 0; k < neuron2_amount; k++)
                            w2_change[j,k] += y1_activity[j] * LearningRate * delta_hidden2[k];
                    }

                    for (int j = 0; j < neuron_amount; j++)
                    {
                        w1_change[j] += training_set[0,i] * LearningRate * delta_hidden1[j];
                    }

                    for (int j = 0; j < neuron2_amount; j++)
                    {
                        w3[j] = w3[j] + w3_change[j];
                        w3_change[j] = 0;
                    }
                    for (int j = 0; j < neuron_amount; j++)
                    {
                        for (int k = 0; k < neuron2_amount; k++)
                        {
                            w2[j, k] = w2[j, k] + w2_change[j, k];
                            w2_change[j, k] = 0;
                        }
                    }
                    for (int j = 0; j < neuron_amount; j++)
                    {
                        w1[j] = w1[j] + w1_change[j];
                        w1_change[j] = 0;
                    }
                    
                    //error = (decimal)Math.Pow((double)delta, 2) / 2;
                    //textBox2.Text += Convert.ToString(error) + "\n\r";
                }
                label3.Text = Convert.ToString(SumSquareError);
                label3.Update();
                RealEpochs++;
                //for (int j = 0; j < neuron_amount; j++)
                //{
                //    w3[j] = w3[j] + w3_change[j] / neuron2_amount;
                //    w1_change[j] = 0;
                //}
                //for (int j = 0; j < neuron_amount; j++)
                //{
                //    for (int k = 0; k < neuron_amount; k++) { 
                //        w2[j, k] = w2[j, k] + w2_change[j, k] / neuron_amount;
                //        w2_change[j, k] = 0;
                //    }
                //}
                //for (int j = 0; j < neuron_amount; j++)
                //{
                //    w1[j] = w1[j] + w1_change[j] / neuron_amount;
                //    w1_change[j] = 0;
                //}
            }
            textBox2.Text = "TRAINING SUCCESS! \n\r epoch : " + Convert.ToString(RealEpochs);
            pub_w.Push(w3);
            pub_w.Push(w2);
            pub_w.Push(w1);

        }

        private void button3_Click(object sender, EventArgs e)
        {
            decimal[] y1 = new decimal[neuron_amount];
            decimal[] y1_activity = new decimal[neuron_amount];
            decimal[] y2 = new decimal[neuron2_amount];
            decimal[] y2_activity = new decimal[neuron2_amount];
            decimal y3, y3_activity;

            decimal[] w1 = new decimal[neuron_amount];
            decimal[,] w2 = new decimal[neuron_amount, neuron2_amount];
            decimal[] w3 = new decimal[neuron2_amount];
            w1 = (decimal[])pub_w.Pop();
            w2 = (decimal[,])pub_w.Pop();
            w3 = (decimal[])pub_w.Pop();

            TestingDataAmount = (int)(4 / 0.01) + 1;
            decimal[] TestingSet = new decimal[TestingDataAmount];
            System.IO.StreamWriter file = new System.IO.StreamWriter(@"C:\Users\ATai\Desktop\BPN\2layer\testing1.txt");
            System.IO.StreamWriter wfile = new System.IO.StreamWriter(@"C:\Users\ATai\Desktop\BPN\2layer\weight.txt");
            string lines;
            for (decimal i = 0; i <= 4; i+=0.01M)
            {
                TestingSet[(int)(i * 100)] = i;
            }



            for(int i = 0; i < TestingDataAmount; i++)
            {

                for (int j = 0; j < neuron_amount; j++)
                {
                    y1[j] = TestingSet[i] * w1[j];
                    y1_activity[j] = 1 / (1 + (decimal)Math.Exp(-(double)y1[j]));
                }

                for (int j = 0; j < neuron2_amount; j++)
                {
                    y2[j] = 0;
                    for (int k = 0; k < neuron_amount; k++)
                        y2[j] += y1_activity[k] * w2[k, j];
                    y2_activity[j] = 1 / (1 + (decimal)Math.Exp(-(double)y2[j])); //激勵y2
                }

                y3 = 0;
                for (int j = 0; j < neuron2_amount; j++)
                    y3 += y2_activity[j] * w3[j];
                y3_activity = 1 / (1 + (decimal)Math.Exp(-(double)y3)); //激勵y3

                lines = Convert.ToString(y3_activity-0.3M);
                file.WriteLine(lines);
            }

            file.WriteLine("========");
            file.WriteLine("=  W1  =");
            file.WriteLine("========");
            for (int i = 0; i < neuron_amount; i++)
                file.WriteLine(Convert.ToString(w1[i]));
            file.WriteLine("========");
            file.WriteLine("=  W2  =");
            file.WriteLine("========");
            for (int i = 0; i < neuron_amount; i++)
                for (int j = 0; j < neuron2_amount; j++)
                    file.WriteLine(Convert.ToString(w2[i,j]));
            file.WriteLine("========");
            file.WriteLine("=  W3  =");
            file.WriteLine("========");
            for (int i = 0; i < neuron_amount; i++)
                file.WriteLine(Convert.ToString(w1[i]));



            file.Close();
            wfile.Close();
            textBox2.Text += "\n\rOUTPUT SUCCESS!";
        }
        
    }
}
