using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NNSharp.DataTypes;

namespace UnitTests
{
    [TestClass]
    public class TestDataArray
    {
        [TestInitialize]
        public void SetUp()
        {
            data = new DataArray(10);
        }

        [TestMethod]
        public void TestMethod_ToZeros()
        {
            data.ToZeros();

            Assert.AreEqual(data[0], 0.0, 0.00000001);
            Assert.AreEqual(data[1], 0.0, 0.00000001);
            Assert.AreEqual(data[2], 0.0, 0.00000001);
        }

        [TestMethod]
        public void TestMethod_ApplyToAll()
        {
            for (int idx = 0; idx < data.GetLength(); ++idx)
            {
              
                data[idx] = 0.0;
                        
            }

            data.ApplyToAll(p => { return p + 1; });

            Assert.AreEqual(data[1], 1.0, 0.00000001);
            Assert.AreEqual(data[3], 1.0, 0.00000001);
            Assert.AreEqual(data[5], 1.0, 0.00000001);
        }

        [TestMethod]
        public void TestMethod_Foreach()
        {
            double sum = 0.0;
            data.ToZeros();

            for (int i = 0; i < 10; ++i)
            {
                data[i] = i + 1;
                sum += i + 1;
            }

            double sumCounted = 0.0;
            foreach(double x in data)
            {
                sumCounted += x;
            }

            Assert.AreEqual(sum, sumCounted, 0.00000001);
        }

        private DataArray data;
    }
}
