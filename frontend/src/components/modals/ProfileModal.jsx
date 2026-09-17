import React, { useState } from 'react';
import { UserRound, Mail } from 'lucide-react';
import { useApp } from '../../context/AppContext';
import { Modal, ModalHeader, TextField } from '../ui/ui';

export default function ProfileModal() {
  const { user, updateProfile, setProfileModalOpen } = useApp();
  const [formData, setFormData] = useState({ name: user.name, email: user.email });

  const handleSubmit = (e) => {
    e.preventDefault();
    updateProfile(formData);
    setProfileModalOpen(false);
  };

  return (
    <Modal onClose={() => setProfileModalOpen(false)}>
      <ModalHeader
        title="Edit profile"
        subtitle="Stored locally on this device."
        onClose={() => setProfileModalOpen(false)}
      />
      <form onSubmit={handleSubmit} className="space-y-4 p-5">
        <div className="flex items-center gap-3.5">
          <span className="t-num flex h-12 w-12 items-center justify-center rounded-xl bg-ink text-[15px] font-semibold text-paper dark:bg-white dark:text-black">
            {formData.name
              .split(' ')
              .map((n) => n[0])
              .join('')
              .toUpperCase()
              .slice(0, 2)}
          </span>
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-ink">{formData.name || 'Your name'}</p>
            <p className="truncate text-xs text-muted">{formData.email || 'email'}</p>
          </div>
        </div>
        <div className="relative">
          <UserRound size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-faint" />
          <input
            name="name"
            value={formData.name}
            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            className="field !pl-10"
            placeholder="Full name"
          />
        </div>
        <div className="relative">
          <Mail size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-faint" />
          <input
            type="email"
            name="email"
            value={formData.email}
            onChange={(e) => setFormData({ ...formData, email: e.target.value })}
            className="field !pl-10"
            placeholder="Email address"
          />
        </div>
        <div className="flex gap-2.5 pt-1">
          <button type="button" onClick={() => setProfileModalOpen(false)} className="btn-ghost flex-1">
            Cancel
          </button>
          <button type="submit" className="btn-primary flex-1">
            Save changes
          </button>
        </div>
      </form>
    </Modal>
  );
}
