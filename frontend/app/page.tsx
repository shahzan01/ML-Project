"use client";
import { useState } from "react";

type Prediction = { predictions: number[] };

export default function Page() {
  const [form, setForm] = useState({
    year: 2016,
    km_driven: 45000,
    model: "Hyundai Verna",
    fuel: "Diesel",
    seller_type: "Individual",
    transmission: "Manual",
    owner: "First Owner",
  });
  const [pred, setPred] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErr(null);
    setPred(null);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items: [form] }),
      });
      const data: Prediction = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(data));
      setPred(data.predictions[0]);
    } catch (e: any) {
      setErr(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const onChange = (k: string, v: any) => setForm((s) => ({ ...s, [k]: v }));

  return (
    <main className="max-w-xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Car Price Predictor</h1>

      <form onSubmit={submit} className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <label className="space-y-1">
            <span>Year</span>
            <input
              type="number"
              className="border rounded p-2 w-full"
              value={form.year}
              onChange={(e) => onChange("year", Number(e.target.value))}
            />
          </label>
          <label className="space-y-1">
            <span>KM Driven</span>
            <input
              type="number"
              className="border rounded p-2 w-full"
              value={form.km_driven}
              onChange={(e) => onChange("km_driven", Number(e.target.value))}
            />
          </label>

          <label className="space-y-1 col-span-2">
            <span>Model (first two words)</span>
            <input
              type="text"
              className="border rounded p-2 w-full"
              value={form.model}
              onChange={(e) => onChange("model", e.target.value)}
            />
          </label>

          <label className="space-y-1">
            <span>Fuel</span>
            <select
              className="border rounded p-2 w-full"
              value={form.fuel}
              onChange={(e) => onChange("fuel", e.target.value)}
            >
              <option>Petrol</option>
              <option>Diesel</option>
              <option>CNG</option>
              <option>LPG</option>
              <option>Electric</option>
            </select>
          </label>

          <label className="space-y-1">
            <span>Seller Type</span>
            <select
              className="border rounded p-2 w-full"
              value={form.seller_type}
              onChange={(e) => onChange("seller_type", e.target.value)}
            >
              <option>Individual</option>
              <option>Dealer</option>
              <option>Trustmark Dealer</option>
            </select>
          </label>

          <label className="space-y-1">
            <span>Transmission</span>
            <select
              className="border rounded p-2 w-full"
              value={form.transmission}
              onChange={(e) => onChange("transmission", e.target.value)}
            >
              <option>Manual</option>
              <option>Automatic</option>
            </select>
          </label>

          <label className="space-y-1">
            <span>Owner</span>
            <select
              className="border rounded p-2 w-full"
              value={form.owner}
              onChange={(e) => onChange("owner", e.target.value)}
            >
              <option>First Owner</option>
              <option>Second Owner</option>
              <option>Third Owner</option>
              <option>Fourth & Above Owner</option>
              <option>Test Drive Car</option>
            </select>
          </label>
        </div>

        <button disabled={loading} className="border rounded px-4 py-2 cursor-pointer ">
          {loading ? "Predicting..." : "Predict Price"}
        </button>
      </form>

      {err && <p className="text-red-600">Error: {err}</p>}
      {pred !== null && (
        <div className="p-4 border rounded">
          <p className="text-lg">
            Estimated Price: ₹ {Math.round(pred).toLocaleString("en-IN")}
          </p>
        </div>
      )}
    </main>
  );
}
