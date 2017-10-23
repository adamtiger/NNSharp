using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.Models;
using NNSharp.SequentialBased.SequentialLayers;
using static NNSharp.DataTypes.Data2D;
using UnitTests.Properties;

namespace UnitTests
{
    [TestClass]
    public class TestBatchNormalization
    {
        [TestMethod]
        public void Test_BatchNorm_Execute()
        {
            // Initialize data.
            Data2D data = new Data2D(2, 1, 3, 4);

            int l = 0;
            for (int b = 0; b < 4; ++b)
            {
                for (int h = 0; h < 2; ++h)
                {
                    for (int w = 0; w < 1; ++w)
                    {
                        for (int c = 0; c < 3; ++c)
                        {
                            l += 1;
                            data[h, w, c, b] = l % 7 - 3;
                        }
                    }
                }
            }

            // Initialize parameters.
            Data2D pms = new Data2D(1, 1, 3, 4);
            pms[0, 0, 0, 0] = 3; // gamma
            pms[0, 0, 1, 0] = 3;
            pms[0, 0, 2, 0] = 3;

            pms[0, 0, 0, 1] = 1; // beta
            pms[0, 0, 1, 1] = 2;
            pms[0, 0, 2, 1] = -1;

            pms[0, 0, 0, 2] = 2; // bias
            pms[0, 0, 1, 2] = 2;
            pms[0, 0, 2, 2] = 2;

            pms[0, 0, 0, 3] = 5; // variance
            pms[0, 0, 1, 3] = 5;
            pms[0, 0, 2, 3] = 5;


            BatchNormLayer bnm = new BatchNormLayer(0.001);
            bnm.SetWeights(pms);
            bnm.SetInput(data);
            bnm.Execute();
            Data2D output = bnm.GetOutput() as Data2D;

            // Checking sizes
            Dimension dim = output.GetDimension();
            Assert.AreEqual(dim.b, 4);
            Assert.AreEqual(dim.c, 3);
            Assert.AreEqual(dim.h, 2);
            Assert.AreEqual(dim.w, 1);

            // Checking calculation
            Assert.AreEqual(output[0, 0, 0, 0], -4.3660264, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 0], -2.02451992, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 0], -3.6830132, 0.000001);
            Assert.AreEqual(output[1, 0, 0, 0], -0.3415066, 0.000001);
            Assert.AreEqual(output[1, 0, 1, 0], 2.0, 0.000001);
            Assert.AreEqual(output[1, 0, 2, 0], 0.34150672, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 1], -5.70753288, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 1], -3.3660264, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 1], -5.02451992, 0.000001);
            Assert.AreEqual(output[1, 0, 0, 1], -1.6830132, 0.000001);
            Assert.AreEqual(output[1, 0, 1, 1], 0.6584934, 0.000001);
            Assert.AreEqual(output[1, 0, 2, 1], -1.0, 0.000001);
            
            Assert.AreEqual(output[0, 0, 0, 2], 2.34150672, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 2], -4.70753288, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 2], -6.3660264, 0.000001);
            Assert.AreEqual(output[1, 0, 0, 2], -3.02451992, 0.000001);
            Assert.AreEqual(output[1, 0, 1, 2], -0.6830132, 0.000001);
            Assert.AreEqual(output[1, 0, 2, 2], -2.34150648, 0.000001);

            Assert.AreEqual(output[0, 0, 0, 3], 1.0, 0.000001);
            Assert.AreEqual(output[0, 0, 1, 3], 3.34150672, 0.000001);
            Assert.AreEqual(output[0, 0, 2, 3], -7.70753288, 0.000001);
            Assert.AreEqual(output[1, 0, 0, 3], -4.3660264, 0.000001);
            Assert.AreEqual(output[1, 0, 1, 3], -2.02451992, 0.000001);
            Assert.AreEqual(output[1, 0, 2, 3], -3.6830132, 0.000001);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_BatchNorm_Null_Input()
        {
            Data2D data = null;
            Data2D weights = new Data2D(1, 1, 3, 3);
            BatchNormLayer bnm = new BatchNormLayer(0.001);
            bnm.SetWeights(weights);
            bnm.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_BatchNorm_Null_Weights()
        {
            Data2D weights = null;
            BatchNormLayer bnm = new BatchNormLayer(0.001);
            bnm.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_BatchNorm_DifferentData_Input()
        {
            DataArray data = new DataArray(5);
            Data2D weights = new Data2D(1, 1, 3, 3);
            BatchNormLayer bnm = new BatchNormLayer(0.001);
            bnm.SetWeights(weights);
            bnm.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_BatchNorm_DifferentData_Weights()
        {
            Data2D weights = new Data2D(1, 2, 3, 3);
            BatchNormLayer bnm = new BatchNormLayer(0.001);
            bnm.SetWeights(weights);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_BatchNorm_WrongSize()
        {
            DataArray weights = new DataArray(5);
            BatchNormLayer bnm = new BatchNormLayer(0.001);
            bnm.SetWeights(weights);
        }

        [TestMethod]
        public void Test_BatchNorm_KerasModel()
        {
            string pathModel = Resources.TestsFolder + "test_batchnorm_model.json";
            string pathInput = Resources.TestsFolder + "test_batchnorm_input.json";
            string pathOutput = Resources.TestsFolder + "test_batchnorm_output.json";

            Utils.KerasModelTest(pathInput, pathModel, pathOutput);
        }  
    }
}

