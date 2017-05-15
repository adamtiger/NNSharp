using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.Models;

namespace UnitTests
{
    [TestClass]
    public class TestSoftmax
    {
        [TestMethod]
        public void Test_Softmax_Execute()
        {

            // Softmax output
            softmax = new SoftmaxLayer();
            Data2D data = new Data2D(1,1,5,1);

            data[0,0,0,0] = 0.0;
            data[0,0,1,0] = 1.0;
            data[0,0,2,0] = 1.5;
            data[0,0,3,0] = 2.0;
            data[0,0,4,0] = 3.0;

            softmax.SetInput(data);
            softmax.Execute();

            Data2D output = softmax.GetOutput() as Data2D;

            // Expected output
            double[] expOu = new double[5];

            double sum = 0.0;
            sum += (Math.Exp(0.0) + Math.Exp(1.0) + Math.Exp(1.5) + Math.Exp(2.0) + Math.Exp(3.0));

            expOu[0] = Math.Exp(0.0) / sum;
            expOu[1] = Math.Exp(1.0) / sum;
            expOu[2] = Math.Exp(1.5) / sum;
            expOu[3] = Math.Exp(2.0) / sum;
            expOu[4] = Math.Exp(3.0) / sum;

            Assert.AreEqual(output[0,0,0,0], expOu[0], 0.00000001);
            Assert.AreEqual(output[0,0,1,0], expOu[1], 0.00000001);
            Assert.AreEqual(output[0,0,2,0], expOu[2], 0.00000001);
            Assert.AreEqual(output[0,0,3,0], expOu[3], 0.00000001);
            Assert.AreEqual(output[0,0,4,0], expOu[4], 0.00000001);
        }

        private SoftmaxLayer softmax;

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Softmax_NullData()
        {
            DataArray data = null;
            SoftmaxLayer soft = new SoftmaxLayer();
            soft.SetInput(data);
        }

        [TestMethod]
        [ExpectedException(typeof(System.Exception))]
        public void Test_Softmax_DifferentData()
        {
            Data2D data = new Data2D(5,4,5,10);
            SoftmaxLayer soft = new SoftmaxLayer();
            soft.SetInput(data);
        }

        [TestMethod]
        public void Test_Softmax_KerasModel()
        {
            string path = @"tests\test_softmax_model.json";
            var reader = new ReaderKerasModel(path);

            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(1, 8, 1, 1);

            inp[0, 0, 0, 0] = 1;
            inp[0, 1, 0, 0] = 2;
            inp[0, 2, 0, 0] = -1;
            inp[0, 3, 0, 0] = 0;

            inp[0, 4, 0, 0] = 3;
            inp[0, 5, 0, 0] = 1;
            inp[0, 6, 0, 0] = 1;
            inp[0, 7, 0, 0] = 2;

            Data2D ou = model.ExecuteNetwork(inp) as Data2D;
            
            Assert.AreEqual(ou.GetDimension().c, 4);
            Assert.AreEqual(ou.GetDimension().w, 1);

            Assert.AreEqual(ou[0, 0, 0, 0], 3.3980058766758248e-09, 1e-10);
            Assert.AreEqual(ou[0, 0, 1, 0], 2.26015504267707e-06, 1e-7);
            Assert.AreEqual(ou[0, 0, 2, 0], 0.9999228715896606, 0.00001);
            Assert.AreEqual(ou[0, 0, 3, 0], 7.484605885110795e-05, 1e-6);
        }
    }
}
