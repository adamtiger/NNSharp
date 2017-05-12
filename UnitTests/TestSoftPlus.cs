using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.SequentialBased.SequentialLayers;
using NNSharp.DataTypes;

namespace UnitTests
{
    [TestClass]
    public class TestSoftPlus
    {
        [TestMethod]
        public void Test_SoftPlus_Execute()
        {
            softplus = new SoftPlusLayer();

            Data2D data = new Data2D(2, 3, 1, 1);
            data[0, 0, 0, 0] = 4;
            data[0, 1, 0, 0] = 2;
            data[0, 2, 0, 0] = -2;

            data[1, 0, 0, 0] = 3;
            data[1, 1, 0, 0] = -1;
            data[1, 2, 0, 0] = -3;

            softplus.SetInput(data);

            softplus.Execute();

            Data2D output = softplus.GetOutput() as Data2D;

            Assert.AreEqual(output[0, 0, 0, 0], SoftPlusFunc(4.0), 0.00000001);
            Assert.AreEqual(output[0, 1, 0, 0], SoftPlusFunc(2.0), 0.00000001);
            Assert.AreEqual(output[0, 2, 0, 0], SoftPlusFunc(-2.0), 0.00000001);

            Assert.AreEqual(output[1, 0, 0, 0], SoftPlusFunc(3.0), 0.00000001);
            Assert.AreEqual(output[1, 1, 0, 0], SoftPlusFunc(-1.0), 0.00000001);
            Assert.AreEqual(output[1, 2, 0, 0], SoftPlusFunc(-3.0), 0.00000001);
        }

        private double SoftPlusFunc(double x)
        {
            return Math.Log(1 + Math.Exp(x));
        }


        private SoftPlusLayer softplus;
    }
}
