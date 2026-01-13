import React, { useState } from 'react';
import { 
  Heart, 
  Apple, 
  Moon, 
  Dumbbell, 
  Brain, 
  Droplets,
  Sun,
  Shield,
  ChevronRight
} from 'lucide-react';

function HealthTips() {
  const [activeCategory, setActiveCategory] = useState('all');

  const categories = [
    { id: 'all', label: 'All Tips', icon: Heart },
    { id: 'nutrition', label: 'Nutrition', icon: Apple },
    { id: 'sleep', label: 'Sleep', icon: Moon },
    { id: 'exercise', label: 'Exercise', icon: Dumbbell },
    { id: 'mental', label: 'Mental Health', icon: Brain },
    { id: 'hydration', label: 'Hydration', icon: Droplets },
  ];

  const healthTips = [
    {
      id: 1,
      category: 'nutrition',
      title: 'Eat a Rainbow of Vegetables',
      description: 'Different colored vegetables contain different nutrients. Aim to include red, orange, yellow, green, and purple vegetables in your daily diet for optimal nutrition.',
      icon: Apple,
      color: 'bg-green-500',
    },
    {
      id: 2,
      category: 'sleep',
      title: 'Maintain a Consistent Sleep Schedule',
      description: 'Go to bed and wake up at the same time every day, even on weekends. This helps regulate your bodys internal clock and improves sleep quality.',
      icon: Moon,
      color: 'bg-indigo-500',
    },
    {
      id: 3,
      category: 'exercise',
      title: '150 Minutes of Weekly Exercise',
      description: 'The WHO recommends at least 150 minutes of moderate-intensity aerobic activity per week. This can include brisk walking, cycling, or swimming.',
      icon: Dumbbell,
      color: 'bg-orange-500',
    },
    {
      id: 4,
      category: 'mental',
      title: 'Practice Mindfulness Daily',
      description: 'Spend 10-15 minutes each day practicing mindfulness or meditation. This can reduce stress, improve focus, and enhance emotional well-being.',
      icon: Brain,
      color: 'bg-purple-500',
    },
    {
      id: 5,
      category: 'hydration',
      title: 'Drink 8 Glasses of Water Daily',
      description: 'Staying hydrated is essential for body temperature regulation, nutrient transport, and organ function. Aim for about 2 liters of water per day.',
      icon: Droplets,
      color: 'bg-blue-500',
    },
    {
      id: 6,
      category: 'nutrition',
      title: 'Reduce Processed Food Intake',
      description: 'Processed foods often contain high levels of sodium, sugar, and unhealthy fats. Choose whole, unprocessed foods whenever possible.',
      icon: Apple,
      color: 'bg-green-500',
    },
    {
      id: 7,
      category: 'sleep',
      title: 'Create a Relaxing Bedtime Routine',
      description: 'Avoid screens for at least an hour before bed. Instead, try reading, gentle stretching, or listening to calming music to prepare for sleep.',
      icon: Moon,
      color: 'bg-indigo-500',
    },
    {
      id: 8,
      category: 'exercise',
      title: 'Include Strength Training',
      description: 'In addition to cardio, include strength training exercises at least twice a week. This helps maintain muscle mass and bone density.',
      icon: Dumbbell,
      color: 'bg-orange-500',
    },
    {
      id: 9,
      category: 'mental',
      title: 'Connect with Others',
      description: 'Social connections are vital for mental health. Make time to connect with friends and family regularly, even if its just a phone call.',
      icon: Brain,
      color: 'bg-purple-500',
    },
    {
      id: 10,
      category: 'hydration',
      title: 'Monitor Your Hydration',
      description: 'Check the color of your urine - pale yellow indicates good hydration. Dark urine may signal you need to drink more water.',
      icon: Droplets,
      color: 'bg-blue-500',
    },
    {
      id: 11,
      category: 'nutrition',
      title: 'Practice Portion Control',
      description: 'Use smaller plates and bowls to help control portion sizes. Eat slowly and stop when you feel satisfied, not stuffed.',
      icon: Apple,
      color: 'bg-green-500',
    },
    {
      id: 12,
      category: 'exercise',
      title: 'Take Regular Movement Breaks',
      description: 'If you sit for long periods, take a 5-minute movement break every hour. Stretch, walk, or do some light exercises.',
      icon: Dumbbell,
      color: 'bg-orange-500',
    },
  ];

  const filteredTips = activeCategory === 'all' 
    ? healthTips 
    : healthTips.filter(tip => tip.category === activeCategory);

  const featuredTip = {
    title: 'Regular Health Check-ups',
    description: 'Prevention is better than cure. Schedule regular health check-ups with your healthcare provider. Early detection of health issues leads to better outcomes. Recommended screenings vary by age and risk factors.',
    icon: Shield,
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <div>
        <h1 className="page-title">Health Tips</h1>
        <p className="text-gray-600 -mt-4 mb-6">
          Evidence-based health tips to help you live a healthier life.
        </p>
      </div>

      <div className="card bg-gradient-to-r from-primary-500 to-blue-600 text-white">
        <div className="flex items-start gap-4">
          <div className="w-14 h-14 bg-white/20 rounded-xl flex items-center justify-center flex-shrink-0">
            <featuredTip.icon size={28} />
          </div>
          <div>
            <span className="text-xs font-medium bg-white/20 px-2 py-1 rounded-full">
              Featured Tip
            </span>
            <h3 className="text-xl font-semibold mt-2">{featuredTip.title}</h3>
            <p className="text-white/80 mt-2">{featuredTip.description}</p>
          </div>
        </div>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
        {categories.map((category) => (
          <button
            key={category.id}
            onClick={() => setActiveCategory(category.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-full font-medium whitespace-nowrap transition-colors ${
              activeCategory === category.id
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            <category.icon size={16} />
            {category.label}
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredTips.map((tip) => (
          <div key={tip.id} className="card hover:shadow-md transition-shadow group">
            <div className="flex items-start gap-4">
              <div className={`w-12 h-12 ${tip.color} rounded-xl flex items-center justify-center flex-shrink-0 text-white`}>
                <tip.icon size={24} />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">
                  {tip.title}
                </h3>
                <p className="text-sm text-gray-500 capitalize mt-1">{tip.category}</p>
              </div>
            </div>
            <p className="text-gray-600 mt-4 text-sm leading-relaxed">
              {tip.description}
            </p>
          </div>
        ))}
      </div>

      <div className="card bg-gradient-to-r from-amber-50 to-orange-50 border-amber-200">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-amber-100 rounded-xl flex items-center justify-center flex-shrink-0">
            <Sun className="text-amber-600" size={24} />
          </div>
          <div className="flex-1">
            <h3 className="font-semibold text-gray-900">Daily Health Reminder</h3>
            <p className="text-sm text-gray-600 mt-1">
              Small, consistent healthy habits lead to significant long-term health benefits. 
              Start with one tip today and gradually build your healthy routine.
            </p>
          </div>
          <ChevronRight className="text-gray-400" size={20} />
        </div>
      </div>
    </div>
  );
}

export default HealthTips;