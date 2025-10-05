import React, { useEffect, useMemo, useState } from 'react';
import {
  StyleSheet, Text, View, Button, Image, Alert, ScrollView,
  TextInput, ActivityIndicator, TouchableOpacity
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';

import * as tf from '@tensorflow/tfjs';
import { Platform } from 'react-native';

if (Platform.OS !== 'web') {
  require('@tensorflow/tfjs-react-native');
} else {
  require('@tensorflow/tfjs-backend-webgl');
}

/* ----------------------------------------------------------------------------
   Helpers
---------------------------------------------------------------------------- */

const MEDIA_TYPE_IMAGES =
  ImagePicker?.MediaType   // new API (Expo SDK 52+)
    ? ImagePicker.MediaType.Images
    : ImagePicker.MediaTypeOptions.Images; // fallback for older SDKs

// Map model labels to pantry names; ignore obvious non-food objects
function cleanLabel(label) {
  const bad = /pan|dish|spatula|packet|bottle|box|carton|plate|bowl|jar|mug|cup|spoon|fork|napkin|bag|stove|microwave|sink|refrigerator|laptop|phone|keyboard/i;
  if (bad.test(label)) return null;

  const m = label.toLowerCase()
    .replace(/ bell pepper.*/,'bell pepper')
    .replace(/ red pepper.*/,'bell pepper')
    .replace(/ green pepper.*/,'bell pepper')
    .replace(/ hotdog|hot dog/,'sausage')
    .replace(/ hamburger|cheeseburger/,'ground beef')
    .replace(/ loaf.*|bread loaf/,'bread')
    .replace(/ spaghetti/,'pasta')
    .replace(/ tomato.*/,'tomato')
    .replace(/ broccoli.*/,'broccoli')
    .replace(/ carrot.*/,'carrot')
    .replace(/ cucumber.*/,'cucumber')
    .replace(/ lemon.*/,'lemon')
    .replace(/ lime.*/,'lime')
    .replace(/ egg.*/,'eggs')
    .replace(/ cheese.*/,'cheese')
    .replace(/ milk.*/,'milk')
    .trim();

  return m.split(/\s+/).slice(0, 2).join(' ');
}

async function ensureCameraPermission() {
  const { status } = await ImagePicker.requestCameraPermissionsAsync();
  return status === 'granted';
}
async function ensureLibraryPermission() {
  const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
  return status === 'granted';
}

/* ----------------------------------------------------------------------------
   Mini recipe set (we’ll expand later)
---------------------------------------------------------------------------- */
const RECIPES = [
  { id:'caprese', title:'Quick Caprese Salad', minutes:10,
    ingredients:['Tomato','Mozzarella','Basil','Olive Oil','Salt','Pepper'],
    steps:['Slice tomatoes & mozzarella.','Layer with basil.','Drizzle olive oil; season.'] },
  { id:'margherita', title:'Margherita Flatbread', minutes:18,
    ingredients:['Flatbread','Tomato','Mozzarella','Basil','Olive Oil','Garlic'],
    steps:['Heat oven 450°F.','Oil+garlic flatbread.','Top with tomato+mozzarella; bake 8–10m.','Finish with basil.'] },
  { id:'omelet', title:'Tomato Basil Omelet', minutes:12,
    ingredients:['Eggs','Tomato','Mozzarella','Basil','Butter','Salt'],
    steps:['Beat eggs.','Cook until almost set.','Add fillings; fold.'] },
  { id:'lemon-chicken', title:'Garlic Lemon Chicken Skillet', minutes:22,
    ingredients:['Chicken Breast','Garlic','Lemon','Olive Oil','Salt','Pepper'],
    steps:['Sear chicken 4–5m/side.','Add garlic 30s.','Add lemon; simmer.'] },
  { id:'stirfry', title:'Weeknight Veggie Stir-Fry', minutes:16,
    ingredients:['Broccoli','Carrot','Bell Pepper','Soy Sauce','Garlic','Ginger','Rice'],
    steps:['Stir-fry veg.','Add garlic+ginger.','Add soy; toss; serve with rice.'] },
  { id:'garlic-broccoli-pasta', title:'Garlic Broccoli Pasta', minutes:20,
    ingredients:['Pasta','Broccoli','Garlic','Olive Oil','Parmesan (optional)'],
    steps:['Boil pasta.','Sauté broccoli+garlic.','Toss with oil+pasta water; finish.'] },
  { id:'rice-bowl', title:'15-Min Rice & Egg Bowl', minutes:15,
    ingredients:['Rice','Eggs','Soy Sauce','Scallion (optional)'],
    steps:['Cook/heat rice.','Fry/soft-scramble eggs.','Top rice with eggs+soy.'] },
  { id:'beans-on-toast', title:'Smoky Beans on Toast', minutes:12,
    ingredients:['Bread','Canned Beans','Tomato Paste (or salsa)','Olive Oil','Garlic','Paprika'],
    steps:['Warm beans with tomato/garlic/paprika.','Serve on toast.'] },
  { id:'tuna-pasta', title:'Pantry Tuna Pasta', minutes:17,
    ingredients:['Pasta','Canned Tuna','Olive Oil','Garlic','Lemon','Parsley (optional)'],
    steps:['Boil pasta.','Sauté garlic in oil.','Add tuna+lemon; toss with pasta.'] },
  { id:'quick-soup', title:'Quick Veg Soup', minutes:20,
    ingredients:['Broth','Carrot','Onion','Celery','Pasta or Rice','Salt','Pepper'],
    steps:['Sauté aromatics.','Add broth+starch; simmer.'] },
  { id:'curry-chickpea', title:'Fast Chickpea Curry', minutes:18,
    ingredients:['Canned Chickpeas','Coconut Milk','Curry Powder','Garlic','Rice'],
    steps:['Sauté curry+garlic.','Add chickpeas+coconut milk; simmer 10m.','Serve with rice.'] },
];

function scoreRecipe(rec, pantryCounts) {
  let hits = 0;
  for (const name of rec.ingredients) {
    const c = pantryCounts.get(name.toLowerCase()) || 0;
    if (c > 0) hits++;
  }
  const base = hits / rec.ingredients.length;
  const speedBonus = rec.minutes <= 20 ? 0.12 : 0;
  return base + speedBonus;
}

/* ----------------------------------------------------------------------------
   App
---------------------------------------------------------------------------- */
export default function App() {
  const [ready, setReady] = useState(false);
  const [model, setModel] = useState(null);

  const [photos, setPhotos] = useState([]);            // { uri }[]
  const [detectedByPhoto, setDetectedByPhoto] = useState([]); // string[]
  const [removed, setRemoved] = useState([]);          // lowercased
  const [added, setAdded] = useState([]);              // lowercased
  const [newItem, setNewItem] = useState('');
  const [recs, setRecs] = useState([]);

  // Load TensorFlow + MobileNet once
  useEffect(() => {
    (async () => {
      try {
        await tf.ready();
        const m = await mobilenet.load({ version: 2, alpha: 0.5 }); // small & fast
        setModel(m);
        setReady(true);
        console.log('✅ Model loaded');
      } catch (e) {
        console.log('TF load error', e);
        Alert.alert('Model error', 'Could not load on-device model. Try reloading.');
      }
    })();
  }, []);

  // Merge detections + your edits into a single count map
  const pantryCounts = useMemo(() => {
    const m = new Map();
    const removedSet = new Set(removed);
    for (const arr of detectedByPhoto) {
      for (const name of arr) {
        const key = name.toLowerCase();
        if (removedSet.has(key)) continue;
        m.set(key, (m.get(key) || 0) + 1);
      }
    }
    for (const key of added) {
      m.set(key, (m.get(key) || 0) + 1);
    }
    return m;
  }, [detectedByPhoto, removed, added]);

  // Classify a URI using base64-js + decodeJpeg (RN safe)
  async function classifyUri(uri) {
    if (!model) throw new Error('Model not ready');

    const b64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' });
    const u8 = base64js.toByteArray(b64);        // Uint8Array
    const imageTensor = decodeJpeg(u8, 3);       // HWC uint8 tensor

    const preds = await model.classify(imageTensor);
    imageTensor.dispose();

    const items = [];
    for (const p of preds) {
      if (p.probability < 0.25) continue;
      const cleaned = cleanLabel(p.className);
      if (cleaned) items.push(cleaned);
    }
    return [...new Set(items)];
  }

  /* ---------- Actions ---------- */

  async function addFromCamera() {
    if (!ready) { Alert.alert('One sec', 'Model still loading…'); return; }
    const ok = await ensureCameraPermission();
    if (!ok) { Alert.alert('Camera permission needed'); return; }

    const result = await ImagePicker.launchCameraAsync({
      quality: 0.9,
      allowsEditing: true,
      mediaTypes: MEDIA_TYPE_IMAGES,
    });
    if (result.canceled) return;

    const uri = result.assets[0].uri;
    setPhotos(p => [...p, { uri }]);

    try {
      const names = await classifyUri(uri);
      setDetectedByPhoto(d => [...d, names]);
    } catch (e) {
      console.log('classify camera error', e);
      setDetectedByPhoto(d => [...d, []]);
    }
  }

  async function addFromPhotos() {
    if (!ready) { Alert.alert('One sec', 'Model still loading…'); return; }
    const ok = await ensureLibraryPermission();
    if (!ok) { Alert.alert('Photos permission needed'); return; }

    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: MEDIA_TYPE_IMAGES,
      allowsMultipleSelection: true,
      quality: 0.9,
      allowsEditing: false,
      selectionLimit: 0, // unlimited on iOS
    });
    if (res.canceled) return;

    for (const a of res.assets) {
      const uri = a.uri;
      setPhotos(p => [...p, { uri }]);
      try {
        const names = await classifyUri(uri);
        setDetectedByPhoto(d => [...d, names]);
      } catch (e) {
        console.log('classify gallery error', e);
        setDetectedByPhoto(d => [...d, []]);
      }
    }
  }

  function removeChip(name) {
    const key = name.toLowerCase();
    setRemoved(r => (r.includes(key) ? r : [...r, key]));
  }
  function addChip() {
    const key = newItem.trim().toLowerCase();
    if (!key) return;
    setAdded(a => [...a, key]);
    setNewItem('');
  }
  function undoEdits() { setRemoved([]); setAdded([]); }
  function clearAll() {
    setPhotos([]); setDetectedByPhoto([]); setRemoved([]); setAdded([]); setRecs([]);
  }

  function computeRecipes() {
    const ranked = RECIPES
      .map(r => ({ ...r, score: scoreRecipe(r, pantryCounts) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 20);
    setRecs(ranked);
  }

  /* ---------- UI ---------- */

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>📸 SnapCook — AI Ingredient Detection</Text>

      {!ready && (
        <View style={{ alignItems: 'center', marginVertical: 12 }}>
          <ActivityIndicator size="large" />
          <Text style={{ marginTop: 8, color: '#64748b' }}>Loading on-device model…</Text>
        </View>
      )}

      <View style={styles.row}>
        <Button title="Add from Camera" onPress={addFromCamera} />
        <View style={{ width: 10 }} />
        <Button title="Add from Photos" onPress={addFromPhotos} />
      </View>

      {photos.length > 0 && (
        <>
          <View style={styles.thumbRow}>
            {photos.map((p, idx) => (
              <Image key={idx} source={{ uri: p.uri }} style={styles.thumb} />
            ))}
          </View>

          <View style={styles.box}>
            <Text style={styles.heading}>Your ingredients (tap ✕ to remove)</Text>
            <View style={styles.chips}>
              {Array.from(pantryCounts.entries()).map(([k, v]) => (
                <View key={k} style={styles.chip}>
                  <Text style={styles.chipText}>{k} ×{v}</Text>
                  <TouchableOpacity onPress={() => removeChip(k)}>
                    <Text style={styles.chipX}>✕</Text>
                  </TouchableOpacity>
                </View>
              ))}
              {Array.from(pantryCounts.entries()).length === 0 && <Text style={styles.text}>—</Text>}
            </View>

            <View style={styles.addRow}>
              <TextInput
                placeholder="Add missing (e.g., chicken, pasta, beans)"
                value={newItem}
                onChangeText={setNewItem}
                style={styles.input}
                autoCapitalize="none"
              />
              <Button title="Add" onPress={addChip} />
            </View>

            {(removed.length > 0 || added.length > 0) && (
              <View style={{ marginTop: 8 }}>
                <Button title="Undo edits" color="#444" onPress={undoEdits} />
              </View>
            )}
          </View>

          <View style={styles.row}>
            <Button title="Find Meal Ideas" onPress={computeRecipes} />
            <View style={{ width: 10 }} />
            <Button title="Clear All" color="#444" onPress={clearAll} />
          </View>
        </>
      )}

      {recs.length > 0 && (
        <View style={styles.box}>
          <Text style={styles.heading}>Recipe ideas (based on your list)</Text>
          {recs.map(r => (
            <View key={r.id} style={styles.card}>
              <Text style={styles.cardTitle}>{r.title}</Text>
              <Text style={styles.meta}>{r.minutes} min</Text>
              <Text style={styles.text}>
                <Text style={styles.bold}>Ingredients:</Text> {r.ingredients.join(', ')}
              </Text>
              <Text style={styles.text}>
                <Text style={styles.bold}>Steps:</Text> {r.steps.join(' ')}
              </Text>
            </View>
          ))}
        </View>
      )}

      <StatusBar style="auto" />
    </ScrollView>
  );
}

/* ----------------------------------------------------------------------------
   Styles
---------------------------------------------------------------------------- */
const styles = StyleSheet.create({
  container: { padding: 16, backgroundColor: '#fff', alignItems: 'center' },
  title: { fontSize: 26, fontWeight: 'bold', marginVertical: 12, textAlign: 'center' },
  row: { flexDirection: 'row', marginBottom: 12 },
  thumbRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginBottom: 8 },
  thumb: { width: 90, height: 90, borderRadius: 10, backgroundColor: '#eee' },
  box: { width: '100%', maxWidth: 680, backgroundColor: '#f8fafc', borderRadius: 12, padding: 12, marginTop: 8 },
  heading: { fontSize: 18, fontWeight: '600', marginBottom: 6 },
  text: { color: '#334155' },
  card: { backgroundColor: '#fff', borderRadius: 10, borderWidth: 1, borderColor: '#e5e7eb', padding: 10, marginTop: 8 },
  cardTitle: { fontWeight: '700', marginBottom: 4 },
  meta: { color: '#64748b', marginBottom: 6 },
  bold: { fontWeight: '700' },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#e2e8f0', borderRadius: 999, paddingHorizontal: 10, paddingVertical: 6 },
  chipText: { marginRight: 6, textTransform: 'capitalize' },
  chipX: { fontWeight: '700' },
  addRow: { flexDirection: 'row', gap: 8, alignItems: 'center', marginTop: 8 },
  input: { flex: 1, borderWidth: 1, borderColor: '#cbd5e1', borderRadius: 8, paddingHorizontal: 10, height: 40, backgroundColor: '#fff' },
});
